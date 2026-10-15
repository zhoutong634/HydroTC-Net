import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset


RAIN_CHANNELS = 36
RAIN_STEP_MINUTES = 5
TIME_STEP_MINUTES = 6
MAX_TIME_MINUTES = 180.0
MAX_WATER_DEPTH = 10.0
RAIN_NORM_FACTOR = 320.0


def pad_tensor_to_multiple(tensor, multiple=16):
    _, h, w = tensor.shape
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple
    return F.pad(tensor, (0, new_w - w, 0, new_h - h))


def load_dem(dem_path, normalize=True):
    dem_raw = np.loadtxt(dem_path, skiprows=6).astype(np.float32)
    valid_mask = dem_raw != -9999
    building_mask = dem_raw == 10
    terrain_mask = valid_mask & (~building_mask)

    dem = dem_raw.copy()

    if normalize:
        if np.any(terrain_mask):
            terrain_values = dem[terrain_mask]
            dem_min = np.min(terrain_values)
            dem_max = np.max(terrain_values)
            dem = (dem - dem_min) / (dem_max - dem_min + 1e-8)
        else:
            dem = np.zeros_like(dem, dtype=np.float32)

    dem[~terrain_mask] = 0.0

    return dem.astype(np.float32), valid_mask, building_mask


def _parse_time_token_to_minutes(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, np.timedelta64):
        return float(value / np.timedelta64(1, "m"))

    text = str(value).strip()
    if not text:
        return np.nan

    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 2:
                hours, minutes = parts
                return float(hours) * 60.0 + float(minutes)
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return float(hours) * 60.0 + float(minutes) + float(seconds) / 60.0
        except ValueError:
            return np.nan

    try:
        return float(text)
    except ValueError:
        return np.nan


def _convert_coord_values_to_minutes(values, units=None):
    values = np.asarray(values)

    if np.issubdtype(values.dtype, np.datetime64):
        return ((values - values[0]) / np.timedelta64(1, "m")).astype(np.float32)

    if np.issubdtype(values.dtype, np.timedelta64):
        return (values / np.timedelta64(1, "m")).astype(np.float32)

    numeric = pd.to_numeric(pd.Series(values.reshape(-1)), errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(numeric).all():
        return None

    units_text = str(units or "").lower()
    scale = 1.0
    if "second" in units_text:
        scale = 1.0 / 60.0
    elif "hour" in units_text:
        scale = 60.0
    elif "day" in units_text:
        scale = 1440.0

    return numeric * scale


def load_rain_series(rain_path, normalize=False, return_times=False, fallback_step_minutes=RAIN_STEP_MINUTES):
    rain_data = pd.read_csv(rain_path, sep=r"\s+", header=None)

    rain_times = None
    value_data = rain_data
    if rain_data.shape[1] > 1:
        parsed_times = rain_data.iloc[:, 0].apply(_parse_time_token_to_minutes)
        if not parsed_times.isna().all():
            rain_times = parsed_times
        value_data = rain_data.iloc[:, 1:]

    value_data = value_data.apply(pd.to_numeric, errors="coerce")
    valid_rows = ~value_data.isna().all(axis=1)
    value_data = value_data.loc[valid_rows].fillna(0)
    rain_series = value_data.to_numpy(dtype=np.float32).reshape(-1)

    if rain_times is not None:
        rain_times = rain_times.loc[valid_rows].to_numpy(dtype=np.float32)
        if value_data.shape[1] > 1:
            rain_times = np.repeat(rain_times, value_data.shape[1]).astype(np.float32)
    else:
        rain_times = (np.arange(len(rain_series), dtype=np.float32) + 1.0) * float(fallback_step_minutes)

    if return_times:
        return rain_series, rain_times
    return rain_series


def load_label(nc_path, time_index):
    with xr.open_dataset(nc_path) as ds:
        return ds["h"][time_index].values


def load_target_time_minutes(nc_path, variable_name="h", fallback_step_minutes=TIME_STEP_MINUTES):
    with xr.open_dataset(nc_path) as ds:
        data_var = ds[variable_name]
        time_count = data_var.shape[0]
        coord_candidates = [data_var.dims[0], "time", "Time", "times", "Times"]

        for coord_name in coord_candidates:
            if coord_name not in ds.coords:
                continue
            coord = ds.coords[coord_name]
            if coord.shape[0] != time_count:
                continue
            minutes = _convert_coord_values_to_minutes(coord.values, coord.attrs.get("units"))
            if minutes is not None and len(minutes) == time_count:
                return minutes.astype(np.float32)

    return np.arange(time_count, dtype=np.float32) * float(fallback_step_minutes)


def select_rain_history(
        rain_series,
        rain_times,
        target_time_minutes,
        rain_channels=RAIN_CHANNELS,
        fallback_step_minutes=RAIN_STEP_MINUTES,
        normalize=True,
        rain_norm_factor=RAIN_NORM_FACTOR):
    rain_series = np.asarray(rain_series, dtype=np.float32)

    if rain_series.size == 0:
        return np.zeros(rain_channels, dtype=np.float32)

    if rain_times is None:
        rain_times = (np.arange(rain_series.size, dtype=np.float32) + 1.0) * float(fallback_step_minutes)
    else:
        rain_times = np.asarray(rain_times, dtype=np.float32)

    history_mask = rain_times <= (float(target_time_minutes) - 1e-6)
    history = rain_series[history_mask]

    if history.size == 0:
        return np.zeros(rain_channels, dtype=np.float32)

    if history.size >= rain_channels:
        selected = history[-rain_channels:]
    else:
        selected = np.pad(history, (rain_channels - history.size, 0), mode="constant")

    if normalize:
        if rain_norm_factor <= 0:
            raise ValueError("rain_norm_factor must be positive")
        selected = selected / float(rain_norm_factor)

    return selected.astype(np.float32)


def build_valid_mask_tensor(valid_mask, building_mask, pad_multiple=16):
    mask_values = (valid_mask.astype(bool) & (~building_mask.astype(bool))).astype(np.float32)
    mask = torch.tensor(mask_values, dtype=torch.float32).unsqueeze(0)
    return pad_tensor_to_multiple(mask, pad_multiple).bool()


def build_flood_tensors(
        rain_series,
        dem,
        label_values,
        time_index,
        rain_times=None,
        target_time_minutes=None,
        normalize=True,
        pad_multiple=16,
        rain_channels=RAIN_CHANNELS,
        time_step_minutes=TIME_STEP_MINUTES,
        max_time_minutes=MAX_TIME_MINUTES,
        max_water_depth=MAX_WATER_DEPTH,
        rain_norm_factor=RAIN_NORM_FACTOR):
    h, w = dem.shape

    if target_time_minutes is None:
        target_time_minutes = time_index * time_step_minutes

    rain_series_fixed = select_rain_history(
        rain_series,
        rain_times,
        target_time_minutes,
        rain_channels=rain_channels,
        normalize=normalize,
        rain_norm_factor=rain_norm_factor
    )

    rain_stack = np.stack(
        [np.full((h, w), rain_value, dtype=np.float32) for rain_value in rain_series_fixed],
        axis=0
    )

    water_time = float(target_time_minutes)
    if normalize:
        water_time = np.clip(water_time / max(max_time_minutes, 1e-8), 0.0, 1.0)
    time_map = np.full((h, w), water_time, dtype=np.float32)

    input_tensor = np.concatenate(
        [
            rain_stack,
            np.expand_dims(dem.astype(np.float32), 0),
            np.expand_dims(time_map, 0),
        ],
        axis=0
    )
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    input_tensor = pad_tensor_to_multiple(input_tensor, pad_multiple)

    label_values = np.nan_to_num(label_values, nan=0.0).astype(np.float32)
    label_values[label_values < 0] = 0.0
    label_values = np.clip(label_values, 0, max_water_depth)
    label = torch.tensor(label_values, dtype=torch.float32).unsqueeze(0)
    label = pad_tensor_to_multiple(label, pad_multiple)

    return input_tensor, label


class MultiEventFloodDataset(Dataset):
    def __init__(self, root_dir, dem_path, normalize=True, pad_multiple=16,
                 rain_norm_factor=RAIN_NORM_FACTOR):
        super().__init__()
        self.root_dir = root_dir
        self.normalize = normalize
        self.pad_multiple = pad_multiple
        self.rain_norm_factor = rain_norm_factor
        self.dem, self.valid_mask, self.building_mask = load_dem(dem_path, normalize=normalize)
        self.samples = []
        self.rain_cache = {}
        self.nc_cache = {}
        self.mask = build_valid_mask_tensor(self.valid_mask, self.building_mask, pad_multiple)

        for event_name in sorted(os.listdir(root_dir)):
            event_path = os.path.join(root_dir, event_name)
            if not os.path.isdir(event_path):
                continue

            rain_path = os.path.join(event_path, "rain.txt")
            nc_path = os.path.join(event_path, "output.nc")
            if not os.path.exists(rain_path) or not os.path.exists(nc_path):
                continue

            target_times = load_target_time_minutes(nc_path)
            time_count = len(target_times)

            for time_index in range(time_count):
                self.samples.append((rain_path, nc_path, time_index, float(target_times[time_index])))

        print(f"self.dem shape: {self.dem.shape}")
        print(f"Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def pad_to_multiple(self, tensor):
        return pad_tensor_to_multiple(tensor, self.pad_multiple)

    def get_rain_data(self, rain_path):
        if rain_path not in self.rain_cache:
            self.rain_cache[rain_path] = load_rain_series(
                rain_path,
                normalize=False,
                return_times=True
            )
        return self.rain_cache[rain_path]

    def get_nc_dataset(self, nc_path):
        if nc_path not in self.nc_cache:
            self.nc_cache[nc_path] = xr.open_dataset(nc_path)
        return self.nc_cache[nc_path]

    def get_label_data(self, nc_path, time_index):
        ds = self.get_nc_dataset(nc_path)
        return ds["h"].isel({ds["h"].dims[0]: time_index}).values

    def close(self):
        for ds in self.nc_cache.values():
            ds.close()
        self.nc_cache.clear()

    def __del__(self):
        self.close()

    def __getitem__(self, idx):
        rain_path, nc_path, time_index, target_time_minutes = self.samples[idx]
        rain_series, rain_times = self.get_rain_data(rain_path)
        label_values = self.get_label_data(nc_path, time_index)

        input_tensor, label = build_flood_tensors(
            rain_series,
            self.dem,
            label_values,
            time_index,
            rain_times=rain_times,
            target_time_minutes=target_time_minutes,
            normalize=self.normalize,
            pad_multiple=self.pad_multiple,
            rain_norm_factor=self.rain_norm_factor
        )
        return input_tensor, label, self.mask
