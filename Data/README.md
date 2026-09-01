# HEEW data contract

Place the same cleaned hourly files used by `2-stages` here:

- `CN03_energy_cleaned.csv`
- `weather_cleaned.csv`

Both files require `Year`, `Month`, `Day`, and `Hour`. The energy file must
contain `Electricity`, `Heat`, `Cooling`, and `PV`. The default `pv10` weather
set requires `Temperature`, `Dew Point`, `Humidity`, `Wind Speed`, `Pressure`,
`Precip`, `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `PV_CLEARNESS_RATIO`, and
`PV_IS_DAYLIGHT`.

Raw datasets and generated experiment outputs are intentionally not committed.

