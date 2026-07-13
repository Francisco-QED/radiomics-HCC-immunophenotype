# Recurrence Follow-Up Gap List

Two distinct groups -- do not conflate them. Both are patients whose recorded follow-up (`time_years`) ends well before their true elapsed time since resection (`Date of surgery`, compared to today), but they carry different implications.

## 1. Genuine 2-year status unknown (n=14) -- chart-review priority

`recur_2yr` is NaN: recorded follow-up ends *before* the 2-year mark, so we do not know whether these patients recurred within 2 years. This is the group that actually affects the 2-year recurrence rate.

| PatientID | Date of surgery | Recorded follow-up (yrs) | True elapsed time (yrs) | Gap (yrs) |
|---|---|---|---|---|
| P168 | 2013-09-09 | 0.54 | 12.84 | 12.30 |
| P177 | 2014-08-27 | 0.13 | 11.87 | 11.75 |
| P073 | 2015-04-29 | 0.46 | 11.20 | 10.74 |
| P184 | 2016-01-07 | 0.85 | 10.51 | 9.66 |
| P123 | 2016-12-22 | 1.14 | 9.55 | 8.41 |
| P096 | 2017-07-13 | 1.31 | 9.00 | 7.69 |
| P153 | 2018-04-05 | 0.63 | 8.27 | 7.64 |
| P065 | 2017-08-03 | 1.34 | 8.94 | 7.60 |
| P079 | 2018-12-13 | 0.65 | 7.58 | 6.93 |
| P012 | 2020-10-29 | 0.07 | 5.70 | 5.63 |
| P118 | 2020-03-17 | 0.97 | 6.32 | 5.35 |
| P159 | 2022-04-28 | 0.23 | 4.21 | 3.97 |
| P150 | 2022-08-18 | 0.73 | 3.90 | 3.17 |
| P187 | 2022-03-24 | 1.54 | 4.30 | 2.77 |

## 2. 2-year status already resolved, longer-term follow-up incomplete (n=24)

`recur_2yr` = 0 (confirmed no recurrence by 2 years) -- **no action needed for the 2-year analysis.** Listed here only because their follow-up beyond 2 years is incomplete, which matters for the full-time-axis Cox-PH/RSF models (C-index, 3yr/5yr landmarks) but not for anything 2-year-specific. Lower priority than group 1.

| PatientID | Date of surgery | Recorded follow-up (yrs) | True elapsed time (yrs) | Gap (yrs) |
|---|---|---|---|---|
| P088 | 2013-12-03 | 2.22 | 12.61 | 10.39 |
| P122 | 2015-03-12 | 2.74 | 11.33 | 8.60 |
| P170 | 2013-12-10 | 4.60 | 12.59 | 7.99 |
| P033 | 2015-05-14 | 3.83 | 11.16 | 7.33 |
| P057 | 2014-09-16 | 5.12 | 11.82 | 6.69 |
| P048 | 2017-03-22 | 2.64 | 9.31 | 6.66 |
| P160 | 2018-06-13 | 2.11 | 8.08 | 5.97 |
| P195 | 2019-03-14 | 3.20 | 7.33 | 4.13 |
| P129 | 2016-04-07 | 6.36 | 10.26 | 3.91 |
| P074 | 2018-08-13 | 4.11 | 7.91 | 3.80 |
| P006 | 2014-05-27 | 8.48 | 12.13 | 3.65 |
| P019 | 2019-02-04 | 3.90 | 7.43 | 3.54 |
| P198 | 2019-03-14 | 4.02 | 7.33 | 3.31 |
| P052 | 2019-04-25 | 4.07 | 7.21 | 3.15 |
| P030 | 2016-05-18 | 7.06 | 10.15 | 3.09 |
| P005 | 2017-03-13 | 6.56 | 9.33 | 2.77 |
| P186 | 2020-09-01 | 3.20 | 5.86 | 2.66 |
| P126 | 2019-03-14 | 5.04 | 7.33 | 2.29 |
| P104 | 2018-12-10 | 5.35 | 7.59 | 2.23 |
| P103 | 2019-04-03 | 5.11 | 7.27 | 2.16 |
| P038 | 2020-02-10 | 4.32 | 6.42 | 2.09 |
| P124 | 2022-03-28 | 2.23 | 4.29 | 2.06 |
| P091 | 2019-09-19 | 4.77 | 6.81 | 2.04 |
| P031 | 2017-11-02 | 6.65 | 8.69 | 2.04 |
