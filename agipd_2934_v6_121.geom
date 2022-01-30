; Geometry file written by EXtra-xwiz
; Geometry used: /gpfs/exfel/exp/SPB/202131/p900215/scratch/dewijnr/geom/agipd_2934_v6.geom
; Format template used: /gpfs/exfel/exp/SPB/202131/p900215/scratch/dewijnr/070820_agipd.geom
; Geometry file written by EXtra-xwiz
; Mills - coursely optimised against p900215/r0433

adu_per_eV = 0.0075  ; no idea
clen = 0.1212
;clen = /CONTROL/SPB_IRU_AGIPD1M/MOTOR/Z_STEPPER/actualPosition/value
;where: h5ls -d /gpfs/exfel/exp/SPB/202001/p002450/proc/r0015/CORR-R0015-DA03-S00000.h5/CONTROL/SPB_IRU_AGIPD1M/MOTOR/Z_STEPPER/actualPosition/value 

photon_energy = 9350
res = 5000 ; 200 um pixels

dim0 = %
dim2 = ss
dim3 = fs

data = /entry_1/instrument_1/detector_1/data

mask = /entry_1/instrument_1/detector_1/mask
mask_good = 0x0000
mask_bad = 0x0001

rigid_group_q0 = p0a0,p0a1,p0a2,p0a3,p0a4,p0a5,p0a6,p0a7,p1a0,p1a1,p1a2,p1a3,p1a4,p1a5,p1a6,p1a7,p2a0,p2a1,p2a2,p2a3,p2a4,p2a5,p2a6,p2a7,p3a0,p3a1,p3a2,p3a3,p3a4,p3a5,p3a6,p3a7
rigid_group_q1 = p4a0,p4a1,p4a2,p4a3,p4a4,p4a5,p4a6,p4a7,p5a0,p5a1,p5a2,p5a3,p5a4,p5a5,p5a6,p5a7,p6a0,p6a1,p6a2,p6a3,p6a4,p6a5,p6a6,p6a7,p7a0,p7a1,p7a2,p7a3,p7a4,p7a5,p7a6,p7a7
rigid_group_q2 = p8a0,p8a1,p8a2,p8a3,p8a4,p8a5,p8a6,p8a7,p9a0,p9a1,p9a2,p9a3,p9a4,p9a5,p9a6,p9a7,p10a0,p10a1,p10a2,p10a3,p10a4,p10a5,p10a6,p10a7,p11a0,p11a1,p11a2,p11a3,p11a4,p11a5,p11a6,p11a7
rigid_group_q3 = p12a0,p12a1,p12a2,p12a3,p12a4,p12a5,p12a6,p12a7,p13a0,p13a1,p13a2,p13a3,p13a4,p13a5,p13a6,p13a7,p14a0,p14a1,p14a2,p14a3,p14a4,p14a5,p14a6,p14a7,p15a0,p15a1,p15a2,p15a3,p15a4,p15a5,p15a6,p15a7

rigid_group_p0 = p0a0,p0a1,p0a2,p0a3,p0a4,p0a5,p0a6,p0a7
rigid_group_p1 = p1a0,p1a1,p1a2,p1a3,p1a4,p1a5,p1a6,p1a7
rigid_group_p2 = p2a0,p2a1,p2a2,p2a3,p2a4,p2a5,p2a6,p2a7
rigid_group_p3 = p3a0,p3a1,p3a2,p3a3,p3a4,p3a5,p3a6,p3a7
rigid_group_p4 = p4a0,p4a1,p4a2,p4a3,p4a4,p4a5,p4a6,p4a7
rigid_group_p5 = p5a0,p5a1,p5a2,p5a3,p5a4,p5a5,p5a6,p5a7
rigid_group_p6 = p6a0,p6a1,p6a2,p6a3,p6a4,p6a5,p6a6,p6a7
rigid_group_p7 = p7a0,p7a1,p7a2,p7a3,p7a4,p7a5,p7a6,p7a7
rigid_group_p8 = p8a0,p8a1,p8a2,p8a3,p8a4,p8a5,p8a6,p8a7
rigid_group_p9 = p9a0,p9a1,p9a2,p9a3,p9a4,p9a5,p9a6,p9a7
rigid_group_p10 = p10a0,p10a1,p10a2,p10a3,p10a4,p10a5,p10a6,p10a7
rigid_group_p11 = p11a0,p11a1,p11a2,p11a3,p11a4,p11a5,p11a6,p11a7
rigid_group_p12 = p12a0,p12a1,p12a2,p12a3,p12a4,p12a5,p12a6,p12a7
rigid_group_p13 = p13a0,p13a1,p13a2,p13a3,p13a4,p13a5,p13a6,p13a7
rigid_group_p14 = p14a0,p14a1,p14a2,p14a3,p14a4,p14a5,p14a6,p14a7
rigid_group_p15 = p15a0,p15a1,p15a2,p15a3,p15a4,p15a5,p15a6,p15a7

rigid_group_collection_quadrants = q0,q1,q2,q3
rigid_group_collection_asics = p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15

p0a0/min_fs = 0
p0a0/min_ss = 0
p0a0/max_fs = 127
p0a0/max_ss = 63
p0a0/fs = -0.002911x -0.999992y
p0a0/ss = +0.999992x -0.002911y
p0a0/corner_x = -536.916
p0a0/corner_y = 619.077

p0a1/min_fs = 0
p0a1/min_ss = 64
p0a1/max_fs = 127
p0a1/max_ss = 127
p0a1/fs = -0.002911x -0.999992y
p0a1/ss = +0.999992x -0.002911y
p0a1/corner_x = -470.916
p0a1/corner_y = 618.864

p0a2/min_fs = 0
p0a2/min_ss = 128
p0a2/max_fs = 127
p0a2/max_ss = 191
p0a2/fs = -0.002911x -0.999992y
p0a2/ss = +0.999992x -0.002911y
p0a2/corner_x = -404.921
p0a2/corner_y = 618.65

p0a3/min_fs = 0
p0a3/min_ss = 192
p0a3/max_fs = 127
p0a3/max_ss = 255
p0a3/fs = -0.002911x -0.999992y
p0a3/ss = +0.999992x -0.002911y
p0a3/corner_x = -338.921
p0a3/corner_y = 618.436

p0a4/min_fs = 0
p0a4/min_ss = 256
p0a4/max_fs = 127
p0a4/max_ss = 319
p0a4/fs = -0.002911x -0.999992y
p0a4/ss = +0.999992x -0.002911y
p0a4/corner_x = -272.922
p0a4/corner_y = 618.221

p0a5/min_fs = 0
p0a5/min_ss = 320
p0a5/max_fs = 127
p0a5/max_ss = 383
p0a5/fs = -0.002911x -0.999992y
p0a5/ss = +0.999992x -0.002911y
p0a5/corner_x = -206.921
p0a5/corner_y = 618.117

p0a6/min_fs = 0
p0a6/min_ss = 384
p0a6/max_fs = 127
p0a6/max_ss = 447
p0a6/fs = -0.002911x -0.999992y
p0a6/ss = +0.999992x -0.002911y
p0a6/corner_x = -140.919
p0a6/corner_y = 617.925

p0a7/min_fs = 0
p0a7/min_ss = 448
p0a7/max_fs = 127
p0a7/max_ss = 511
p0a7/fs = -0.002911x -0.999992y
p0a7/ss = +0.999992x -0.002911y
p0a7/corner_x = -74.9212
p0a7/corner_y = 617.733

p1a0/min_fs = 0
p1a0/min_ss = 0
p1a0/max_fs = 127
p1a0/max_ss = 63
p1a0/fs = -0.004581x -0.999991y
p1a0/ss = +0.999991x -0.004581y
p1a0/corner_x = -537.558
p1a0/corner_y = 462.5

p1a1/min_fs = 0
p1a1/min_ss = 64
p1a1/max_fs = 127
p1a1/max_ss = 127
p1a1/fs = -0.004581x -0.999991y
p1a1/ss = +0.999991x -0.004581y
p1a1/corner_x = -471.56
p1a1/corner_y = 462.21

p1a2/min_fs = 0
p1a2/min_ss = 128
p1a2/max_fs = 127
p1a2/max_ss = 191
p1a2/fs = -0.004581x -0.999991y
p1a2/ss = +0.999991x -0.004581y
p1a2/corner_x = -405.564
p1a2/corner_y = 461.92

p1a3/min_fs = 0
p1a3/min_ss = 192
p1a3/max_fs = 127
p1a3/max_ss = 255
p1a3/fs = -0.004581x -0.999991y
p1a3/ss = +0.999991x -0.004581y
p1a3/corner_x = -339.567
p1a3/corner_y = 461.627

p1a4/min_fs = 0
p1a4/min_ss = 256
p1a4/max_fs = 127
p1a4/max_ss = 319
p1a4/fs = -0.004581x -0.999991y
p1a4/ss = +0.999991x -0.004581y
p1a4/corner_x = -273.568
p1a4/corner_y = 461.341

p1a5/min_fs = 0
p1a5/min_ss = 320
p1a5/max_fs = 127
p1a5/max_ss = 383
p1a5/fs = -0.004581x -0.999991y
p1a5/ss = +0.999991x -0.004581y
p1a5/corner_x = -207.574
p1a5/corner_y = 461.048

p1a6/min_fs = 0
p1a6/min_ss = 384
p1a6/max_fs = 127
p1a6/max_ss = 447
p1a6/fs = -0.004581x -0.999991y
p1a6/ss = +0.999991x -0.004581y
p1a6/corner_x = -141.559
p1a6/corner_y = 460.757

p1a7/min_fs = 0
p1a7/min_ss = 448
p1a7/max_fs = 127
p1a7/max_ss = 511
p1a7/fs = -0.004581x -0.999991y
p1a7/ss = +0.999991x -0.004581y
p1a7/corner_x = -75.5576
p1a7/corner_y = 460.467

p2a0/min_fs = 0
p2a0/min_ss = 0
p2a0/max_fs = 127
p2a0/max_ss = 63
p2a0/fs = -0.003931x -0.999994y
p2a0/ss = +0.999994x -0.003931y
p2a0/corner_x = -538.035
p2a0/corner_y = 305.745

p2a1/min_fs = 0
p2a1/min_ss = 64
p2a1/max_fs = 127
p2a1/max_ss = 127
p2a1/fs = -0.003931x -0.999994y
p2a1/ss = +0.999994x -0.003931y
p2a1/corner_x = -472.036
p2a1/corner_y = 305.478

p2a2/min_fs = 0
p2a2/min_ss = 128
p2a2/max_fs = 127
p2a2/max_ss = 191
p2a2/fs = -0.003931x -0.999994y
p2a2/ss = +0.999994x -0.003931y
p2a2/corner_x = -406.036
p2a2/corner_y = 305.212

p2a3/min_fs = 0
p2a3/min_ss = 192
p2a3/max_fs = 127
p2a3/max_ss = 255
p2a3/fs = -0.003931x -0.999994y
p2a3/ss = +0.999994x -0.003931y
p2a3/corner_x = -340.037
p2a3/corner_y = 304.942

p2a4/min_fs = 0
p2a4/min_ss = 256
p2a4/max_fs = 127
p2a4/max_ss = 319
p2a4/fs = -0.003931x -0.999994y
p2a4/ss = +0.999994x -0.003931y
p2a4/corner_x = -274.039
p2a4/corner_y = 304.674

p2a5/min_fs = 0
p2a5/min_ss = 320
p2a5/max_fs = 127
p2a5/max_ss = 383
p2a5/fs = -0.003931x -0.999994y
p2a5/ss = +0.999994x -0.003931y
p2a5/corner_x = -208.043
p2a5/corner_y = 304.409

p2a6/min_fs = 0
p2a6/min_ss = 384
p2a6/max_fs = 127
p2a6/max_ss = 447
p2a6/fs = -0.003931x -0.999994y
p2a6/ss = +0.999994x -0.003931y
p2a6/corner_x = -142.028
p2a6/corner_y = 304.139

p2a7/min_fs = 0
p2a7/min_ss = 448
p2a7/max_fs = 127
p2a7/max_ss = 511
p2a7/fs = -0.003931x -0.999994y
p2a7/ss = +0.999994x -0.003931y
p2a7/corner_x = -76.0275
p2a7/corner_y = 303.87

p3a0/min_fs = 0
p3a0/min_ss = 0
p3a0/max_fs = 127
p3a0/max_ss = 63
p3a0/fs = -0.003898x -0.999991y
p3a0/ss = +0.999991x -0.003898y
p3a0/corner_x = -539.257
p3a0/corner_y = 148.854

p3a1/min_fs = 0
p3a1/min_ss = 64
p3a1/max_fs = 127
p3a1/max_ss = 127
p3a1/fs = -0.003898x -0.999991y
p3a1/ss = +0.999991x -0.003898y
p3a1/corner_x = -473.261
p3a1/corner_y = 148.601

p3a2/min_fs = 0
p3a2/min_ss = 128
p3a2/max_fs = 127
p3a2/max_ss = 191
p3a2/fs = -0.003898x -0.999991y
p3a2/ss = +0.999991x -0.003898y
p3a2/corner_x = -407.262
p3a2/corner_y = 148.343

p3a3/min_fs = 0
p3a3/min_ss = 192
p3a3/max_fs = 127
p3a3/max_ss = 255
p3a3/fs = -0.003898x -0.999991y
p3a3/ss = +0.999991x -0.003898y
p3a3/corner_x = -341.264
p3a3/corner_y = 148.087

p3a4/min_fs = 0
p3a4/min_ss = 256
p3a4/max_fs = 127
p3a4/max_ss = 319
p3a4/fs = -0.003898x -0.999991y
p3a4/ss = +0.999991x -0.003898y
p3a4/corner_x = -275.267
p3a4/corner_y = 147.831

p3a5/min_fs = 0
p3a5/min_ss = 320
p3a5/max_fs = 127
p3a5/max_ss = 383
p3a5/fs = -0.003898x -0.999991y
p3a5/ss = +0.999991x -0.003898y
p3a5/corner_x = -209.252
p3a5/corner_y = 147.576

p3a6/min_fs = 0
p3a6/min_ss = 384
p3a6/max_fs = 127
p3a6/max_ss = 447
p3a6/fs = -0.003898x -0.999991y
p3a6/ss = +0.999991x -0.003898y
p3a6/corner_x = -143.251
p3a6/corner_y = 147.322

p3a7/min_fs = 0
p3a7/min_ss = 448
p3a7/max_fs = 127
p3a7/max_ss = 511
p3a7/fs = -0.003898x -0.999991y
p3a7/ss = +0.999991x -0.003898y
p3a7/corner_x = -77.2512
p3a7/corner_y = 147.066

p4a0/min_fs = 0
p4a0/min_ss = 0
p4a0/max_fs = 127
p4a0/max_ss = 63
p4a0/fs = -0.005389x -0.999987y
p4a0/ss = +0.999987x -0.005389y
p4a0/corner_x = -543.229
p4a0/corner_y = -21.6255

p4a1/min_fs = 0
p4a1/min_ss = 64
p4a1/max_fs = 127
p4a1/max_ss = 127
p4a1/fs = -0.005389x -0.999987y
p4a1/ss = +0.999987x -0.005389y
p4a1/corner_x = -477.234
p4a1/corner_y = -21.9727

p4a2/min_fs = 0
p4a2/min_ss = 128
p4a2/max_fs = 127
p4a2/max_ss = 191
p4a2/fs = -0.005389x -0.999987y
p4a2/ss = +0.999987x -0.005389y
p4a2/corner_x = -411.236
p4a2/corner_y = -22.3202

p4a3/min_fs = 0
p4a3/min_ss = 192
p4a3/max_fs = 127
p4a3/max_ss = 255
p4a3/fs = -0.005389x -0.999987y
p4a3/ss = +0.999987x -0.005389y
p4a3/corner_x = -345.239
p4a3/corner_y = -22.6676

p4a4/min_fs = 0
p4a4/min_ss = 256
p4a4/max_fs = 127
p4a4/max_ss = 319
p4a4/fs = -0.005389x -0.999987y
p4a4/ss = +0.999987x -0.005389y
p4a4/corner_x = -279.241
p4a4/corner_y = -23.0149

p4a5/min_fs = 0
p4a5/min_ss = 320
p4a5/max_fs = 127
p4a5/max_ss = 383
p4a5/fs = -0.005389x -0.999987y
p4a5/ss = +0.999987x -0.005389y
p4a5/corner_x = -213.243
p4a5/corner_y = -23.3624

p4a6/min_fs = 0
p4a6/min_ss = 384
p4a6/max_fs = 127
p4a6/max_ss = 447
p4a6/fs = -0.005389x -0.999987y
p4a6/ss = +0.999987x -0.005389y
p4a6/corner_x = -147.246
p4a6/corner_y = -23.7096

p4a7/min_fs = 0
p4a7/min_ss = 448
p4a7/max_fs = 127
p4a7/max_ss = 511
p4a7/fs = -0.005389x -0.999987y
p4a7/ss = +0.999987x -0.005389y
p4a7/corner_x = -81.2533
p4a7/corner_y = -24.0571

p5a0/min_fs = 0
p5a0/min_ss = 0
p5a0/max_fs = 127
p5a0/max_ss = 63
p5a0/fs = -0.003025x -0.999994y
p5a0/ss = +0.999994x -0.003025y
p5a0/corner_x = -544.321
p5a0/corner_y = -180.382

p5a1/min_fs = 0
p5a1/min_ss = 64
p5a1/max_fs = 127
p5a1/max_ss = 127
p5a1/fs = -0.003025x -0.999994y
p5a1/ss = +0.999994x -0.003025y
p5a1/corner_x = -478.325
p5a1/corner_y = -180.587

p5a2/min_fs = 0
p5a2/min_ss = 128
p5a2/max_fs = 127
p5a2/max_ss = 191
p5a2/fs = -0.003025x -0.999994y
p5a2/ss = +0.999994x -0.003025y
p5a2/corner_x = -412.327
p5a2/corner_y = -180.795

p5a3/min_fs = 0
p5a3/min_ss = 192
p5a3/max_fs = 127
p5a3/max_ss = 255
p5a3/fs = -0.003025x -0.999994y
p5a3/ss = +0.999994x -0.003025y
p5a3/corner_x = -346.328
p5a3/corner_y = -181.003

p5a4/min_fs = 0
p5a4/min_ss = 256
p5a4/max_fs = 127
p5a4/max_ss = 319
p5a4/fs = -0.003025x -0.999994y
p5a4/ss = +0.999994x -0.003025y
p5a4/corner_x = -280.33
p5a4/corner_y = -181.211

p5a5/min_fs = 0
p5a5/min_ss = 320
p5a5/max_fs = 127
p5a5/max_ss = 383
p5a5/fs = -0.003025x -0.999994y
p5a5/ss = +0.999994x -0.003025y
p5a5/corner_x = -214.316
p5a5/corner_y = -181.421

p5a6/min_fs = 0
p5a6/min_ss = 384
p5a6/max_fs = 127
p5a6/max_ss = 447
p5a6/fs = -0.003025x -0.999994y
p5a6/ss = +0.999994x -0.003025y
p5a6/corner_x = -148.316
p5a6/corner_y = -181.627

p5a7/min_fs = 0
p5a7/min_ss = 448
p5a7/max_fs = 127
p5a7/max_ss = 511
p5a7/fs = -0.003025x -0.999994y
p5a7/ss = +0.999994x -0.003025y
p5a7/corner_x = -82.3149
p5a7/corner_y = -181.833

p6a0/min_fs = 0
p6a0/min_ss = 0
p6a0/max_fs = 127
p6a0/max_ss = 63
p6a0/fs = -0.006805x -0.999979y
p6a0/ss = +0.999979x -0.006805y
p6a0/corner_x = -544.887
p6a0/corner_y = -335.951

p6a1/min_fs = 0
p6a1/min_ss = 64
p6a1/max_fs = 127
p6a1/max_ss = 127
p6a1/fs = -0.006805x -0.999979y
p6a1/ss = +0.999979x -0.006805y
p6a1/corner_x = -478.89
p6a1/corner_y = -336.404

p6a2/min_fs = 0
p6a2/min_ss = 128
p6a2/max_fs = 127
p6a2/max_ss = 191
p6a2/fs = -0.006805x -0.999979y
p6a2/ss = +0.999979x -0.006805y
p6a2/corner_x = -412.895
p6a2/corner_y = -336.865

p6a3/min_fs = 0
p6a3/min_ss = 192
p6a3/max_fs = 127
p6a3/max_ss = 255
p6a3/fs = -0.006805x -0.999979y
p6a3/ss = +0.999979x -0.006805y
p6a3/corner_x = -346.898
p6a3/corner_y = -337.322

p6a4/min_fs = 0
p6a4/min_ss = 256
p6a4/max_fs = 127
p6a4/max_ss = 319
p6a4/fs = -0.006805x -0.999979y
p6a4/ss = +0.999979x -0.006805y
p6a4/corner_x = -280.904
p6a4/corner_y = -337.781

p6a5/min_fs = 0
p6a5/min_ss = 320
p6a5/max_fs = 127
p6a5/max_ss = 383
p6a5/fs = -0.006805x -0.999979y
p6a5/ss = +0.999979x -0.006805y
p6a5/corner_x = -214.907
p6a5/corner_y = -338.239

p6a6/min_fs = 0
p6a6/min_ss = 384
p6a6/max_fs = 127
p6a6/max_ss = 447
p6a6/fs = -0.006805x -0.999979y
p6a6/ss = +0.999979x -0.006805y
p6a6/corner_x = -148.912
p6a6/corner_y = -338.697

p6a7/min_fs = 0
p6a7/min_ss = 448
p6a7/max_fs = 127
p6a7/max_ss = 511
p6a7/fs = -0.006805x -0.999979y
p6a7/ss = +0.999979x -0.006805y
p6a7/corner_x = -82.9119
p6a7/corner_y = -339.155

p7a0/min_fs = 0
p7a0/min_ss = 0
p7a0/max_fs = 127
p7a0/max_ss = 63
p7a0/fs = -0.003913x -0.999991y
p7a0/ss = +0.999991x -0.003913y
p7a0/corner_x = -544.848
p7a0/corner_y = -493.155

p7a1/min_fs = 0
p7a1/min_ss = 64
p7a1/max_fs = 127
p7a1/max_ss = 127
p7a1/fs = -0.003913x -0.999991y
p7a1/ss = +0.999991x -0.003913y
p7a1/corner_x = -478.855
p7a1/corner_y = -493.415

p7a2/min_fs = 0
p7a2/min_ss = 128
p7a2/max_fs = 127
p7a2/max_ss = 191
p7a2/fs = -0.003913x -0.999991y
p7a2/ss = +0.999991x -0.003913y
p7a2/corner_x = -412.859
p7a2/corner_y = -493.675

p7a3/min_fs = 0
p7a3/min_ss = 192
p7a3/max_fs = 127
p7a3/max_ss = 255
p7a3/fs = -0.003913x -0.999991y
p7a3/ss = +0.999991x -0.003913y
p7a3/corner_x = -346.862
p7a3/corner_y = -493.937

p7a4/min_fs = 0
p7a4/min_ss = 256
p7a4/max_fs = 127
p7a4/max_ss = 319
p7a4/fs = -0.003913x -0.999991y
p7a4/ss = +0.999991x -0.003913y
p7a4/corner_x = -280.866
p7a4/corner_y = -494.199

p7a5/min_fs = 0
p7a5/min_ss = 320
p7a5/max_fs = 127
p7a5/max_ss = 383
p7a5/fs = -0.003913x -0.999991y
p7a5/ss = +0.999991x -0.003913y
p7a5/corner_x = -214.87
p7a5/corner_y = -494.46

p7a6/min_fs = 0
p7a6/min_ss = 384
p7a6/max_fs = 127
p7a6/max_ss = 447
p7a6/fs = -0.003913x -0.999991y
p7a6/ss = +0.999991x -0.003913y
p7a6/corner_x = -148.87
p7a6/corner_y = -494.719

p7a7/min_fs = 0
p7a7/min_ss = 448
p7a7/max_fs = 127
p7a7/max_ss = 511
p7a7/fs = -0.003913x -0.999991y
p7a7/ss = +0.999991x -0.003913y
p7a7/corner_x = -82.8756
p7a7/corner_y = -494.98

p8a0/min_fs = 0
p8a0/min_ss = 0
p8a0/max_fs = 127
p8a0/max_ss = 63
p8a0/fs = +0.003402x +0.999994y
p8a0/ss = -0.999994x +0.003402y
p8a0/corner_x = 532.289
p8a0/corner_y = -161.39

p8a1/min_fs = 0
p8a1/min_ss = 64
p8a1/max_fs = 127
p8a1/max_ss = 127
p8a1/fs = +0.003402x +0.999994y
p8a1/ss = -0.999994x +0.003402y
p8a1/corner_x = 466.292
p8a1/corner_y = -161.171

p8a2/min_fs = 0
p8a2/min_ss = 128
p8a2/max_fs = 127
p8a2/max_ss = 191
p8a2/fs = +0.003402x +0.999994y
p8a2/ss = -0.999994x +0.003402y
p8a2/corner_x = 400.293
p8a2/corner_y = -160.95

p8a3/min_fs = 0
p8a3/min_ss = 192
p8a3/max_fs = 127
p8a3/max_ss = 255
p8a3/fs = +0.003402x +0.999994y
p8a3/ss = -0.999994x +0.003402y
p8a3/corner_x = 334.296
p8a3/corner_y = -160.729

p8a4/min_fs = 0
p8a4/min_ss = 256
p8a4/max_fs = 127
p8a4/max_ss = 319
p8a4/fs = +0.003402x +0.999994y
p8a4/ss = -0.999994x +0.003402y
p8a4/corner_x = 268.298
p8a4/corner_y = -160.508

p8a5/min_fs = 0
p8a5/min_ss = 320
p8a5/max_fs = 127
p8a5/max_ss = 383
p8a5/fs = +0.003402x +0.999994y
p8a5/ss = -0.999994x +0.003402y
p8a5/corner_x = 202.3
p8a5/corner_y = -160.292

p8a6/min_fs = 0
p8a6/min_ss = 384
p8a6/max_fs = 127
p8a6/max_ss = 447
p8a6/fs = +0.003402x +0.999994y
p8a6/ss = -0.999994x +0.003402y
p8a6/corner_x = 136.302
p8a6/corner_y = -160.071

p8a7/min_fs = 0
p8a7/min_ss = 448
p8a7/max_fs = 127
p8a7/max_ss = 511
p8a7/fs = +0.003402x +0.999994y
p8a7/ss = -0.999994x +0.003402y
p8a7/corner_x = 70.3063
p8a7/corner_y = -159.853

p9a0/min_fs = 0
p9a0/min_ss = 0
p9a0/max_fs = 127
p9a0/max_ss = 63
p9a0/fs = +0.004632x +0.999989y
p9a0/ss = -0.999989x +0.004632y
p9a0/corner_x = 531.456
p9a0/corner_y = -318.808

p9a1/min_fs = 0
p9a1/min_ss = 64
p9a1/max_fs = 127
p9a1/max_ss = 127
p9a1/fs = +0.004632x +0.999989y
p9a1/ss = -0.999989x +0.004632y
p9a1/corner_x = 465.459
p9a1/corner_y = -318.505

p9a2/min_fs = 0
p9a2/min_ss = 128
p9a2/max_fs = 127
p9a2/max_ss = 191
p9a2/fs = +0.004632x +0.999989y
p9a2/ss = -0.999989x +0.004632y
p9a2/corner_x = 399.462
p9a2/corner_y = -318.204

p9a3/min_fs = 0
p9a3/min_ss = 192
p9a3/max_fs = 127
p9a3/max_ss = 255
p9a3/fs = +0.004632x +0.999989y
p9a3/ss = -0.999989x +0.004632y
p9a3/corner_x = 333.466
p9a3/corner_y = -317.9

p9a4/min_fs = 0
p9a4/min_ss = 256
p9a4/max_fs = 127
p9a4/max_ss = 319
p9a4/fs = +0.004632x +0.999989y
p9a4/ss = -0.999989x +0.004632y
p9a4/corner_x = 267.469
p9a4/corner_y = -317.6

p9a5/min_fs = 0
p9a5/min_ss = 320
p9a5/max_fs = 127
p9a5/max_ss = 383
p9a5/fs = +0.004632x +0.999989y
p9a5/ss = -0.999989x +0.004632y
p9a5/corner_x = 201.473
p9a5/corner_y = -317.299

p9a6/min_fs = 0
p9a6/min_ss = 384
p9a6/max_fs = 127
p9a6/max_ss = 447
p9a6/fs = +0.004632x +0.999989y
p9a6/ss = -0.999989x +0.004632y
p9a6/corner_x = 135.455
p9a6/corner_y = -316.993

p9a7/min_fs = 0
p9a7/min_ss = 448
p9a7/max_fs = 127
p9a7/max_ss = 511
p9a7/fs = +0.004632x +0.999989y
p9a7/ss = -0.999989x +0.004632y
p9a7/corner_x = 69.4534
p9a7/corner_y = -316.692

p10a0/min_fs = 0
p10a0/min_ss = 0
p10a0/max_fs = 127
p10a0/max_ss = 63
p10a0/fs = +0.003180x +0.999994y
p10a0/ss = -0.999994x +0.003180y
p10a0/corner_x = 530.894
p10a0/corner_y = -475.635

p10a1/min_fs = 0
p10a1/min_ss = 64
p10a1/max_fs = 127
p10a1/max_ss = 127
p10a1/fs = +0.003180x +0.999994y
p10a1/ss = -0.999994x +0.003180y
p10a1/corner_x = 464.899
p10a1/corner_y = -475.437

p10a2/min_fs = 0
p10a2/min_ss = 128
p10a2/max_fs = 127
p10a2/max_ss = 191
p10a2/fs = +0.003180x +0.999994y
p10a2/ss = -0.999994x +0.003180y
p10a2/corner_x = 398.901
p10a2/corner_y = -475.234

p10a3/min_fs = 0
p10a3/min_ss = 192
p10a3/max_fs = 127
p10a3/max_ss = 255
p10a3/fs = +0.003180x +0.999994y
p10a3/ss = -0.999994x +0.003180y
p10a3/corner_x = 332.905
p10a3/corner_y = -475.034

p10a4/min_fs = 0
p10a4/min_ss = 256
p10a4/max_fs = 127
p10a4/max_ss = 319
p10a4/fs = +0.003180x +0.999994y
p10a4/ss = -0.999994x +0.003180y
p10a4/corner_x = 266.908
p10a4/corner_y = -474.83

p10a5/min_fs = 0
p10a5/min_ss = 320
p10a5/max_fs = 127
p10a5/max_ss = 383
p10a5/fs = +0.003180x +0.999994y
p10a5/ss = -0.999994x +0.003180y
p10a5/corner_x = 200.91
p10a5/corner_y = -474.629

p10a6/min_fs = 0
p10a6/min_ss = 384
p10a6/max_fs = 127
p10a6/max_ss = 447
p10a6/fs = +0.003180x +0.999994y
p10a6/ss = -0.999994x +0.003180y
p10a6/corner_x = 134.915
p10a6/corner_y = -474.43

p10a7/min_fs = 0
p10a7/min_ss = 448
p10a7/max_fs = 127
p10a7/max_ss = 511
p10a7/fs = +0.003180x +0.999994y
p10a7/ss = -0.999994x +0.003180y
p10a7/corner_x = 68.9179
p10a7/corner_y = -474.229

p11a0/min_fs = 0
p11a0/min_ss = 0
p11a0/max_fs = 127
p11a0/max_ss = 63
p11a0/fs = +0.000680x +1.000000y
p11a0/ss = -1.000000x +0.000680y
p11a0/corner_x = 530.428
p11a0/corner_y = -630.478

p11a1/min_fs = 0
p11a1/min_ss = 64
p11a1/max_fs = 127
p11a1/max_ss = 127
p11a1/fs = +0.000680x +1.000000y
p11a1/ss = -1.000000x +0.000680y
p11a1/corner_x = 464.427
p11a1/corner_y = -630.438

p11a2/min_fs = 0
p11a2/min_ss = 128
p11a2/max_fs = 127
p11a2/max_ss = 191
p11a2/fs = +0.000680x +1.000000y
p11a2/ss = -1.000000x +0.000680y
p11a2/corner_x = 398.431
p11a2/corner_y = -630.399

p11a3/min_fs = 0
p11a3/min_ss = 192
p11a3/max_fs = 127
p11a3/max_ss = 255
p11a3/fs = +0.000680x +1.000000y
p11a3/ss = -1.000000x +0.000680y
p11a3/corner_x = 332.432
p11a3/corner_y = -630.358

p11a4/min_fs = 0
p11a4/min_ss = 256
p11a4/max_fs = 127
p11a4/max_ss = 319
p11a4/fs = +0.000680x +1.000000y
p11a4/ss = -1.000000x +0.000680y
p11a4/corner_x = 266.436
p11a4/corner_y = -630.319

p11a5/min_fs = 0
p11a5/min_ss = 320
p11a5/max_fs = 127
p11a5/max_ss = 383
p11a5/fs = +0.000680x +1.000000y
p11a5/ss = -1.000000x +0.000680y
p11a5/corner_x = 200.437
p11a5/corner_y = -630.284

p11a6/min_fs = 0
p11a6/min_ss = 384
p11a6/max_fs = 127
p11a6/max_ss = 447
p11a6/fs = +0.000680x +1.000000y
p11a6/ss = -1.000000x +0.000680y
p11a6/corner_x = 134.441
p11a6/corner_y = -630.242

p11a7/min_fs = 0
p11a7/min_ss = 448
p11a7/max_fs = 127
p11a7/max_ss = 511
p11a7/fs = +0.000680x +1.000000y
p11a7/ss = -1.000000x +0.000680y
p11a7/corner_x = 68.4415
p11a7/corner_y = -630.204

p12a0/min_fs = 0
p12a0/min_ss = 0
p12a0/max_fs = 127
p12a0/max_ss = 63
p12a0/fs = -0.000319x +1.000001y
p12a0/ss = -1.000001x -0.000319y
p12a0/corner_x = 540.869
p12a0/corner_y = 489.265

p12a1/min_fs = 0
p12a1/min_ss = 64
p12a1/max_fs = 127
p12a1/max_ss = 127
p12a1/fs = -0.000319x +1.000001y
p12a1/ss = -1.000001x -0.000319y
p12a1/corner_x = 474.869
p12a1/corner_y = 489.198

p12a2/min_fs = 0
p12a2/min_ss = 128
p12a2/max_fs = 127
p12a2/max_ss = 191
p12a2/fs = -0.000319x +1.000001y
p12a2/ss = -1.000001x -0.000319y
p12a2/corner_x = 408.873
p12a2/corner_y = 489.128

p12a3/min_fs = 0
p12a3/min_ss = 192
p12a3/max_fs = 127
p12a3/max_ss = 255
p12a3/fs = -0.000319x +1.000001y
p12a3/ss = -1.000001x -0.000319y
p12a3/corner_x = 342.87
p12a3/corner_y = 489.202

p12a4/min_fs = 0
p12a4/min_ss = 256
p12a4/max_fs = 127
p12a4/max_ss = 319
p12a4/fs = -0.000319x +1.000001y
p12a4/ss = -1.000001x -0.000319y
p12a4/corner_x = 276.867
p12a4/corner_y = 489.18

p12a5/min_fs = 0
p12a5/min_ss = 320
p12a5/max_fs = 127
p12a5/max_ss = 383
p12a5/fs = -0.000319x +1.000001y
p12a5/ss = -1.000001x -0.000319y
p12a5/corner_x = 210.868
p12a5/corner_y = 489.157

p12a6/min_fs = 0
p12a6/min_ss = 384
p12a6/max_fs = 127
p12a6/max_ss = 447
p12a6/fs = -0.000319x +1.000001y
p12a6/ss = -1.000001x -0.000319y
p12a6/corner_x = 144.869
p12a6/corner_y = 489.14

p12a7/min_fs = 0
p12a7/min_ss = 448
p12a7/max_fs = 127
p12a7/max_ss = 511
p12a7/fs = -0.000319x +1.000001y
p12a7/ss = -1.000001x -0.000319y
p12a7/corner_x = 78.8686
p12a7/corner_y = 489.115

p13a0/min_fs = 0
p13a0/min_ss = 0
p13a0/max_fs = 127
p13a0/max_ss = 63
p13a0/fs = +0.005264x +0.999987y
p13a0/ss = -0.999987x +0.005264y
p13a0/corner_x = 537.968
p13a0/corner_y = 330.1

p13a1/min_fs = 0
p13a1/min_ss = 64
p13a1/max_fs = 127
p13a1/max_ss = 127
p13a1/fs = +0.005264x +0.999987y
p13a1/ss = -0.999987x +0.005264y
p13a1/corner_x = 471.971
p13a1/corner_y = 330.432

p13a2/min_fs = 0
p13a2/min_ss = 128
p13a2/max_fs = 127
p13a2/max_ss = 191
p13a2/fs = +0.005264x +0.999987y
p13a2/ss = -0.999987x +0.005264y
p13a2/corner_x = 405.978
p13a2/corner_y = 330.771

p13a3/min_fs = 0
p13a3/min_ss = 192
p13a3/max_fs = 127
p13a3/max_ss = 255
p13a3/fs = +0.005264x +0.999987y
p13a3/ss = -0.999987x +0.005264y
p13a3/corner_x = 339.981
p13a3/corner_y = 331.105

p13a4/min_fs = 0
p13a4/min_ss = 256
p13a4/max_fs = 127
p13a4/max_ss = 319
p13a4/fs = +0.005264x +0.999987y
p13a4/ss = -0.999987x +0.005264y
p13a4/corner_x = 273.984
p13a4/corner_y = 331.439

p13a5/min_fs = 0
p13a5/min_ss = 320
p13a5/max_fs = 127
p13a5/max_ss = 383
p13a5/fs = +0.005264x +0.999987y
p13a5/ss = -0.999987x +0.005264y
p13a5/corner_x = 207.969
p13a5/corner_y = 331.775

p13a6/min_fs = 0
p13a6/min_ss = 384
p13a6/max_fs = 127
p13a6/max_ss = 447
p13a6/fs = +0.005264x +0.999987y
p13a6/ss = -0.999987x +0.005264y
p13a6/corner_x = 141.967
p13a6/corner_y = 332.115

p13a7/min_fs = 0
p13a7/min_ss = 448
p13a7/max_fs = 127
p13a7/max_ss = 511
p13a7/fs = +0.005264x +0.999987y
p13a7/ss = -0.999987x +0.005264y
p13a7/corner_x = 75.9718
p13a7/corner_y = 332.447

p14a0/min_fs = 0
p14a0/min_ss = 0
p14a0/max_fs = 127
p14a0/max_ss = 63
p14a0/fs = +0.002274x +0.999997y
p14a0/ss = -0.999997x +0.002274y
p14a0/corner_x = 538.619
p14a0/corner_y = 173.969

p14a1/min_fs = 0
p14a1/min_ss = 64
p14a1/max_fs = 127
p14a1/max_ss = 127
p14a1/fs = +0.002274x +0.999997y
p14a1/ss = -0.999997x +0.002274y
p14a1/corner_x = 472.621
p14a1/corner_y = 174.117

p14a2/min_fs = 0
p14a2/min_ss = 128
p14a2/max_fs = 127
p14a2/max_ss = 191
p14a2/fs = +0.002274x +0.999997y
p14a2/ss = -0.999997x +0.002274y
p14a2/corner_x = 406.625
p14a2/corner_y = 174.262

p14a3/min_fs = 0
p14a3/min_ss = 192
p14a3/max_fs = 127
p14a3/max_ss = 255
p14a3/fs = +0.002274x +0.999997y
p14a3/ss = -0.999997x +0.002274y
p14a3/corner_x = 340.627
p14a3/corner_y = 174.409

p14a4/min_fs = 0
p14a4/min_ss = 256
p14a4/max_fs = 127
p14a4/max_ss = 319
p14a4/fs = +0.002274x +0.999997y
p14a4/ss = -0.999997x +0.002274y
p14a4/corner_x = 274.63
p14a4/corner_y = 174.556

p14a5/min_fs = 0
p14a5/min_ss = 320
p14a5/max_fs = 127
p14a5/max_ss = 383
p14a5/fs = +0.002274x +0.999997y
p14a5/ss = -0.999997x +0.002274y
p14a5/corner_x = 208.616
p14a5/corner_y = 174.703

p14a6/min_fs = 0
p14a6/min_ss = 384
p14a6/max_fs = 127
p14a6/max_ss = 447
p14a6/fs = +0.002274x +0.999997y
p14a6/ss = -0.999997x +0.002274y
p14a6/corner_x = 142.616
p14a6/corner_y = 174.85

p14a7/min_fs = 0
p14a7/min_ss = 448
p14a7/max_fs = 127
p14a7/max_ss = 511
p14a7/fs = +0.002274x +0.999997y
p14a7/ss = -0.999997x +0.002274y
p14a7/corner_x = 76.6129
p14a7/corner_y = 174.998

p15a0/min_fs = 0
p15a0/min_ss = 0
p15a0/max_fs = 127
p15a0/max_ss = 63
p15a0/fs = +0.004210x +0.999994y
p15a0/ss = -0.999994x +0.004210y
p15a0/corner_x = 537.74
p15a0/corner_y = 16.5757

p15a1/min_fs = 0
p15a1/min_ss = 64
p15a1/max_fs = 127
p15a1/max_ss = 127
p15a1/fs = +0.004210x +0.999994y
p15a1/ss = -0.999994x +0.004210y
p15a1/corner_x = 471.743
p15a1/corner_y = 16.8473

p15a2/min_fs = 0
p15a2/min_ss = 128
p15a2/max_fs = 127
p15a2/max_ss = 191
p15a2/fs = +0.004210x +0.999994y
p15a2/ss = -0.999994x +0.004210y
p15a2/corner_x = 405.748
p15a2/corner_y = 17.1189

p15a3/min_fs = 0
p15a3/min_ss = 192
p15a3/max_fs = 127
p15a3/max_ss = 255
p15a3/fs = +0.004210x +0.999994y
p15a3/ss = -0.999994x +0.004210y
p15a3/corner_x = 339.75
p15a3/corner_y = 17.3906

p15a4/min_fs = 0
p15a4/min_ss = 256
p15a4/max_fs = 127
p15a4/max_ss = 319
p15a4/fs = +0.004210x +0.999994y
p15a4/ss = -0.999994x +0.004210y
p15a4/corner_x = 273.751
p15a4/corner_y = 17.6624

p15a5/min_fs = 0
p15a5/min_ss = 320
p15a5/max_fs = 127
p15a5/max_ss = 383
p15a5/fs = +0.004210x +0.999994y
p15a5/ss = -0.999994x +0.004210y
p15a5/corner_x = 207.735
p15a5/corner_y = 17.9341

p15a6/min_fs = 0
p15a6/min_ss = 384
p15a6/max_fs = 127
p15a6/max_ss = 447
p15a6/fs = +0.004210x +0.999994y
p15a6/ss = -0.999994x +0.004210y
p15a6/corner_x = 141.734
p15a6/corner_y = 18.2058

p15a7/min_fs = 0
p15a7/min_ss = 448
p15a7/max_fs = 127
p15a7/max_ss = 511
p15a7/fs = +0.004210x +0.999994y
p15a7/ss = -0.999994x +0.004210y
p15a7/corner_x = 75.7353
p15a7/corner_y = 18.4775















p0a0/dim1 = 0
p0a1/dim1 = 0
p0a2/dim1 = 0
p0a3/dim1 = 0
p0a4/dim1 = 0
p0a5/dim1 = 0
p0a6/dim1 = 0
p0a7/dim1 = 0
p1a0/dim1 = 1
p1a1/dim1 = 1
p1a2/dim1 = 1
p1a3/dim1 = 1
p1a4/dim1 = 1
p1a5/dim1 = 1
p1a6/dim1 = 1
p1a7/dim1 = 1
p2a0/dim1 = 2
p2a1/dim1 = 2
p2a2/dim1 = 2
p2a3/dim1 = 2
p2a4/dim1 = 2
p2a5/dim1 = 2
p2a6/dim1 = 2
p2a7/dim1 = 2
p3a0/dim1 = 3
p3a1/dim1 = 3
p3a2/dim1 = 3
p3a3/dim1 = 3
p3a4/dim1 = 3
p3a5/dim1 = 3
p3a6/dim1 = 3
p3a7/dim1 = 3
p4a0/dim1 = 4
p4a1/dim1 = 4
p4a2/dim1 = 4
p4a3/dim1 = 4
p4a4/dim1 = 4
p4a5/dim1 = 4
p4a6/dim1 = 4
p4a7/dim1 = 4
p5a0/dim1 = 5
p5a1/dim1 = 5
p5a2/dim1 = 5
p5a3/dim1 = 5
p5a4/dim1 = 5
p5a5/dim1 = 5
p5a6/dim1 = 5
p5a7/dim1 = 5
p6a0/dim1 = 6
p6a1/dim1 = 6
p6a2/dim1 = 6
p6a3/dim1 = 6
p6a4/dim1 = 6
p6a5/dim1 = 6
p6a6/dim1 = 6
p6a7/dim1 = 6
p7a0/dim1 = 7
p7a1/dim1 = 7
p7a2/dim1 = 7
p7a3/dim1 = 7
p7a4/dim1 = 7
p7a5/dim1 = 7
p7a6/dim1 = 7
p7a7/dim1 = 7
p8a0/dim1 = 8
p8a1/dim1 = 8
p8a2/dim1 = 8
p8a3/dim1 = 8
p8a4/dim1 = 8
p8a5/dim1 = 8
p8a6/dim1 = 8
p8a7/dim1 = 8
p9a0/dim1 = 9
p9a1/dim1 = 9
p9a2/dim1 = 9
p9a3/dim1 = 9
p9a4/dim1 = 9
p9a5/dim1 = 9
p9a6/dim1 = 9
p9a7/dim1 = 9
p10a0/dim1 = 10
p10a1/dim1 = 10
p10a2/dim1 = 10
p10a3/dim1 = 10
p10a4/dim1 = 10
p10a5/dim1 = 10
p10a6/dim1 = 10
p10a7/dim1 = 10
p11a0/dim1 = 11
p11a1/dim1 = 11
p11a2/dim1 = 11
p11a3/dim1 = 11
p11a4/dim1 = 11
p11a5/dim1 = 11
p11a6/dim1 = 11
p11a7/dim1 = 11
p12a0/dim1 = 12
p12a1/dim1 = 12
p12a2/dim1 = 12
p12a3/dim1 = 12
p12a4/dim1 = 12
p12a5/dim1 = 12
p12a6/dim1 = 12
p12a7/dim1 = 12
p13a0/dim1 = 13
p13a1/dim1 = 13
p13a2/dim1 = 13
p13a3/dim1 = 13
p13a4/dim1 = 13
p13a5/dim1 = 13
p13a6/dim1 = 13
p13a7/dim1 = 13
p14a0/dim1 = 14
p14a1/dim1 = 14
p14a2/dim1 = 14
p14a3/dim1 = 14
p14a4/dim1 = 14
p14a5/dim1 = 14
p14a6/dim1 = 14
p14a7/dim1 = 14
p15a0/dim1 = 15
p15a1/dim1 = 15
p15a2/dim1 = 15
p15a3/dim1 = 15
p15a4/dim1 = 15
p15a5/dim1 = 15
p15a6/dim1 = 15
p15a7/dim1 = 15


p0a0/dim2 = ss
p0a1/dim2 = ss
p0a2/dim2 = ss
p0a3/dim2 = ss
p0a4/dim2 = ss
p0a5/dim2 = ss
p0a6/dim2 = ss
p0a7/dim2 = ss
p1a0/dim2 = ss
p1a1/dim2 = ss
p1a2/dim2 = ss
p1a3/dim2 = ss
p1a4/dim2 = ss
p1a5/dim2 = ss
p1a6/dim2 = ss
p1a7/dim2 = ss
p2a0/dim2 = ss
p2a1/dim2 = ss
p2a2/dim2 = ss
p2a3/dim2 = ss
p2a4/dim2 = ss
p2a5/dim2 = ss
p2a6/dim2 = ss
p2a7/dim2 = ss
p3a0/dim2 = ss
p3a1/dim2 = ss
p3a2/dim2 = ss
p3a3/dim2 = ss
p3a4/dim2 = ss
p3a5/dim2 = ss
p3a6/dim2 = ss
p3a7/dim2 = ss
p4a0/dim2 = ss
p4a1/dim2 = ss
p4a2/dim2 = ss
p4a3/dim2 = ss
p4a4/dim2 = ss
p4a5/dim2 = ss
p4a6/dim2 = ss
p4a7/dim2 = ss
p5a0/dim2 = ss
p5a1/dim2 = ss
p5a2/dim2 = ss
p5a3/dim2 = ss
p5a4/dim2 = ss
p5a5/dim2 = ss
p5a6/dim2 = ss
p5a7/dim2 = ss
p6a0/dim2 = ss
p6a1/dim2 = ss
p6a2/dim2 = ss
p6a3/dim2 = ss
p6a4/dim2 = ss
p6a5/dim2 = ss
p6a6/dim2 = ss
p6a7/dim2 = ss
p7a0/dim2 = ss
p7a1/dim2 = ss
p7a2/dim2 = ss
p7a3/dim2 = ss
p7a4/dim2 = ss
p7a5/dim2 = ss
p7a6/dim2 = ss
p7a7/dim2 = ss
p8a0/dim2 = ss
p8a1/dim2 = ss
p8a2/dim2 = ss
p8a3/dim2 = ss
p8a4/dim2 = ss
p8a5/dim2 = ss
p8a6/dim2 = ss
p8a7/dim2 = ss
p9a0/dim2 = ss
p9a1/dim2 = ss
p9a2/dim2 = ss
p9a3/dim2 = ss
p9a4/dim2 = ss
p9a5/dim2 = ss
p9a6/dim2 = ss
p9a7/dim2 = ss
p10a0/dim2 = ss
p10a1/dim2 = ss
p10a2/dim2 = ss
p10a3/dim2 = ss
p10a4/dim2 = ss
p10a5/dim2 = ss
p10a6/dim2 = ss
p10a7/dim2 = ss
p11a0/dim2 = ss
p11a1/dim2 = ss
p11a2/dim2 = ss
p11a3/dim2 = ss
p11a4/dim2 = ss
p11a5/dim2 = ss
p11a6/dim2 = ss
p11a7/dim2 = ss
p12a0/dim2 = ss
p12a1/dim2 = ss
p12a2/dim2 = ss
p12a3/dim2 = ss
p12a4/dim2 = ss
p12a5/dim2 = ss
p12a6/dim2 = ss
p12a7/dim2 = ss
p13a0/dim2 = ss
p13a1/dim2 = ss
p13a2/dim2 = ss
p13a3/dim2 = ss
p13a4/dim2 = ss
p13a5/dim2 = ss
p13a6/dim2 = ss
p13a7/dim2 = ss
p14a0/dim2 = ss
p14a1/dim2 = ss
p14a2/dim2 = ss
p14a3/dim2 = ss
p14a4/dim2 = ss
p14a5/dim2 = ss
p14a6/dim2 = ss
p14a7/dim2 = ss
p15a0/dim2 = ss
p15a1/dim2 = ss
p15a2/dim2 = ss
p15a3/dim2 = ss
p15a4/dim2 = ss
p15a5/dim2 = ss
p15a6/dim2 = ss
p15a7/dim2 = ss


p0a0/dim3 = fs
p0a1/dim3 = fs
p0a2/dim3 = fs
p0a3/dim3 = fs
p0a4/dim3 = fs
p0a5/dim3 = fs
p0a6/dim3 = fs
p0a7/dim3 = fs
p1a0/dim3 = fs
p1a1/dim3 = fs
p1a2/dim3 = fs
p1a3/dim3 = fs
p1a4/dim3 = fs
p1a5/dim3 = fs
p1a6/dim3 = fs
p1a7/dim3 = fs
p2a0/dim3 = fs
p2a1/dim3 = fs
p2a2/dim3 = fs
p2a3/dim3 = fs
p2a4/dim3 = fs
p2a5/dim3 = fs
p2a6/dim3 = fs
p2a7/dim3 = fs
p3a0/dim3 = fs
p3a1/dim3 = fs
p3a2/dim3 = fs
p3a3/dim3 = fs
p3a4/dim3 = fs
p3a5/dim3 = fs
p3a6/dim3 = fs
p3a7/dim3 = fs
p4a0/dim3 = fs
p4a1/dim3 = fs
p4a2/dim3 = fs
p4a3/dim3 = fs
p4a4/dim3 = fs
p4a5/dim3 = fs
p4a6/dim3 = fs
p4a7/dim3 = fs
p5a0/dim3 = fs
p5a1/dim3 = fs
p5a2/dim3 = fs
p5a3/dim3 = fs
p5a4/dim3 = fs
p5a5/dim3 = fs
p5a6/dim3 = fs
p5a7/dim3 = fs
p6a0/dim3 = fs
p6a1/dim3 = fs
p6a2/dim3 = fs
p6a3/dim3 = fs
p6a4/dim3 = fs
p6a5/dim3 = fs
p6a6/dim3 = fs
p6a7/dim3 = fs
p7a0/dim3 = fs
p7a1/dim3 = fs
p7a2/dim3 = fs
p7a3/dim3 = fs
p7a4/dim3 = fs
p7a5/dim3 = fs
p7a6/dim3 = fs
p7a7/dim3 = fs
p8a0/dim3 = fs
p8a1/dim3 = fs
p8a2/dim3 = fs
p8a3/dim3 = fs
p8a4/dim3 = fs
p8a5/dim3 = fs
p8a6/dim3 = fs
p8a7/dim3 = fs
p9a0/dim3 = fs
p9a1/dim3 = fs
p9a2/dim3 = fs
p9a3/dim3 = fs
p9a4/dim3 = fs
p9a5/dim3 = fs
p9a6/dim3 = fs
p9a7/dim3 = fs
p10a0/dim3 = fs
p10a1/dim3 = fs
p10a2/dim3 = fs
p10a3/dim3 = fs
p10a4/dim3 = fs
p10a5/dim3 = fs
p10a6/dim3 = fs
p10a7/dim3 = fs
p11a0/dim3 = fs
p11a1/dim3 = fs
p11a2/dim3 = fs
p11a3/dim3 = fs
p11a4/dim3 = fs
p11a5/dim3 = fs
p11a6/dim3 = fs
p11a7/dim3 = fs
p12a0/dim3 = fs
p12a1/dim3 = fs
p12a2/dim3 = fs
p12a3/dim3 = fs
p12a4/dim3 = fs
p12a5/dim3 = fs
p12a6/dim3 = fs
p12a7/dim3 = fs
p13a0/dim3 = fs
p13a1/dim3 = fs
p13a2/dim3 = fs
p13a3/dim3 = fs
p13a4/dim3 = fs
p13a5/dim3 = fs
p13a6/dim3 = fs
p13a7/dim3 = fs
p14a0/dim3 = fs
p14a1/dim3 = fs
p14a2/dim3 = fs
p14a3/dim3 = fs
p14a4/dim3 = fs
p14a5/dim3 = fs
p14a6/dim3 = fs
p14a7/dim3 = fs
p15a0/dim3 = fs
p15a1/dim3 = fs
p15a2/dim3 = fs
p15a3/dim3 = fs
p15a4/dim3 = fs
p15a5/dim3 = fs
p15a6/dim3 = fs
p15a7/dim3 = fs


p0a0/coffset = 0.000672
p0a1/coffset = 0.000672
p0a2/coffset = 0.000672
p0a3/coffset = 0.000672
p0a4/coffset = 0.000672
p0a5/coffset = 0.000672
p0a6/coffset = 0.000672
p0a7/coffset = 0.000672
p1a0/coffset = 0.000511
p1a1/coffset = 0.000511
p1a2/coffset = 0.000511
p1a3/coffset = 0.000511
p1a4/coffset = 0.000511
p1a5/coffset = 0.000511
p1a6/coffset = 0.000511
p1a7/coffset = 0.000511
p2a0/coffset = 0.000197
p2a1/coffset = 0.000197
p2a2/coffset = 0.000197
p2a3/coffset = 0.000197
p2a4/coffset = 0.000197
p2a5/coffset = 0.000197
p2a6/coffset = 0.000197
p2a7/coffset = 0.000197
p3a0/coffset = 0.000414
p3a1/coffset = 0.000414
p3a2/coffset = 0.000414
p3a3/coffset = 0.000414
p3a4/coffset = 0.000414
p3a5/coffset = 0.000414
p3a6/coffset = 0.000414
p3a7/coffset = 0.000414
p4a0/coffset = 0.000161
p4a1/coffset = 0.000161
p4a2/coffset = 0.000161
p4a3/coffset = 0.000161
p4a4/coffset = 0.000161
p4a5/coffset = 0.000161
p4a6/coffset = 0.000161
p4a7/coffset = 0.000161
p5a0/coffset = 0.000214
p5a1/coffset = 0.000214
p5a2/coffset = 0.000214
p5a3/coffset = 0.000214
p5a4/coffset = 0.000214
p5a5/coffset = 0.000214
p5a6/coffset = 0.000214
p5a7/coffset = 0.000214
p6a0/coffset = 0.000245
p6a1/coffset = 0.000245
p6a2/coffset = 0.000245
p6a3/coffset = 0.000245
p6a4/coffset = 0.000245
p6a5/coffset = 0.000245
p6a6/coffset = 0.000245
p6a7/coffset = 0.000245
p7a0/coffset = 0.000242
p7a1/coffset = 0.000242
p7a2/coffset = 0.000242
p7a3/coffset = 0.000242
p7a4/coffset = 0.000242
p7a5/coffset = 0.000242
p7a6/coffset = 0.000242
p7a7/coffset = 0.000242
p8a0/coffset = 0.000013
p8a1/coffset = 0.000013
p8a2/coffset = 0.000013
p8a3/coffset = 0.000013
p8a4/coffset = 0.000013
p8a5/coffset = 0.000013
p8a6/coffset = 0.000013
p8a7/coffset = 0.000013
p9a0/coffset = 0.000140
p9a1/coffset = 0.000140
p9a2/coffset = 0.000140
p9a3/coffset = 0.000140
p9a4/coffset = 0.000140
p9a5/coffset = 0.000140
p9a6/coffset = 0.000140
p9a7/coffset = 0.000140
p10a0/coffset = 0.000286
p10a1/coffset = 0.000286
p10a2/coffset = 0.000286
p10a3/coffset = 0.000286
p10a4/coffset = 0.000286
p10a5/coffset = 0.000286
p10a6/coffset = 0.000286
p10a7/coffset = 0.000286
p11a0/coffset = -0.000027
p11a1/coffset = -0.000027
p11a2/coffset = -0.000027
p11a3/coffset = -0.000027
p11a4/coffset = -0.000027
p11a5/coffset = -0.000027
p11a6/coffset = -0.000027
p11a7/coffset = -0.000027
p12a0/coffset = 0.000601
p12a1/coffset = 0.000601
p12a2/coffset = 0.000601
p12a3/coffset = 0.000601
p12a4/coffset = 0.000601
p12a5/coffset = 0.000601
p12a6/coffset = 0.000601
p12a7/coffset = 0.000601
p13a0/coffset = 0.000382
p13a1/coffset = 0.000382
p13a2/coffset = 0.000382
p13a3/coffset = 0.000382
p13a4/coffset = 0.000382
p13a5/coffset = 0.000382
p13a6/coffset = 0.000382
p13a7/coffset = 0.000382
p14a0/coffset = 0.000270
p14a1/coffset = 0.000270
p14a2/coffset = 0.000270
p14a3/coffset = 0.000270
p14a4/coffset = 0.000270
p14a5/coffset = 0.000270
p14a6/coffset = 0.000270
p14a7/coffset = 0.000270
p15a0/coffset = -0.000032
p15a1/coffset = -0.000032
p15a2/coffset = -0.000032
p15a3/coffset = -0.000032
p15a4/coffset = -0.000032
p15a5/coffset = -0.000032
p15a6/coffset = -0.000032
p15a7/coffset = -0.000032

