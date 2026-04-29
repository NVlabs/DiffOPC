# Best params

Failed:

- aim-0408-hparams

______________________________________________________________________

Conclusion:

**StepSize == 4 is the best.**
**max_iter == 60**
Overall, the exp can be very fast if no sraf edges.
We can insert sraf before the exp. using the 256 size.

when do sraf, the step size can be slower.

______________________________________________________________________

steps=(1 2 4 8)
iters=(60 70 80 90)
opc.SEG_LENGTH=60 80 100

run.experiment == 'iter_seg60'  & 'iter_seg60_sraf"

best opc iter = 80 , best step size = 4

min epe = 3.2
min l2 = 30119
min pvb = 51924
min shot = 76
min iter

run.experiment == 'iter_seg100"
min runtime = 4.87s

______________________________________________________________________

aim_0410_pvb_weight

WeightPVBL2=0.9
min epe = 21

______________________________________________________________________

Runtime

512

```log
[2024-04-11 21:11:18,499][__main__][INFO] - [Testcase 1]: L2 43310; PVBand 56413; EPE 5; Shot: 105; BestIter: 132 SolveTime: 17.85s
[2024-04-11 21:11:18,499][__main__][INFO] - Logging hyperparameters!
[2024-04-11 21:11:18,539][__main__][INFO] - avg_l2: 43310.0; avg_pvb: 56413.0; avg_epe: 5.0; avg_shot: 105.0; avg_runtime: 17.8;
[2024-04-11 21:11:18,540][src.utils.utils][INFO] - Output dir: /home/local/eda13/gc29434/phd/intern/DiffOPC/logs/mscale/runs/2024-04-11_21-10-51
```

single epoch = 17.85 / 140

2048

```log
[2024-04-11 21:07:44,680][__main__][INFO] - [Testcase 1]: L2 38483; PVBand 55896; EPE 4; Shot: 131; BestIter: 139 SolveTime: 29.26s
[2024-04-11 21:07:44,680][__main__][INFO] - Logging hyperparameters!
[2024-04-11 21:07:44,720][__main__][INFO] - avg_l2: 38483.0; avg_pvb: 55896.0; avg_epe: 4.0; avg_shot: 131.0; avg_runtime: 29.3;
```
