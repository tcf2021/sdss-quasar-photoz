# SDSS 类星体测光红移估计 V2

本项目使用固定的 SDSS DR16Q_v4 数据快照，对比 kNN、随机森林、XGBoost 和 PyTorch MLP。实验包括四组测光特征、三折交叉验证、独立的新测试集、异常对象及颜色近邻分析、控制训练样本量的亮度偏移实验。所有性能数字由实际运行生成。

## 1. 与 V1 的关系

- V1 保留在原目录，V2 是独立项目目录，不修改旧代码、模型、数据或 TEST_OPENED.json。
- V2 的开发集来自 V1 的训练集与验证集合并，按你的正式实验为 170,000 条。
- 新测试集默认 30,000 条，从原始目录中本次 V1 全部 200,000 条之外抽取。
- 排除对象时使用整个目录上 SDSS_NAME / 正 THING_ID 的关联身份分组，包含经中间记录建立的重复关系。仅换一个随机种子重新切那 20 万条不构成新测试集。
- 读取原始 FITS 时核对 V1 记录的 SHA256，并核对原划分、对象 ID、标签和测光值。旧测试集仅用于身份排除，不参与开发训练。
- 新测试集在任何 V2 调参之前冻结。首次 evaluate 写入 V2 的 TEST_OPENED.json，此后训练被锁定，只允许同一冻结选择重评估和重载验证。

## 2. Windows：从已有环境开始

目标目录结构：

```text
D:\AI\photoz-windows\sdss-photoz       原 V1、.venv-win、data/raw、data/processed
D:\AI\photoz-windows\sdss-photoz-v2    本项目
```

在 Windows PowerShell 中解压下载的 ZIP：

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\sdss-photoz-v2.zip" -DestinationPath 'D:\AI\photoz-windows'
cd D:\AI\photoz-windows\sdss-photoz-v2
```

逐条执行，每一步显示 Stage completed 后再进行下一步：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage install
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage check -Device cuda
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage prepare
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage smoke -Device cuda
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage all -Device cuda
```

install 复用原 .venv-win，仅安装固定的非 Torch 依赖，保留你已有的 PyTorch 2.9.1+cu130。check 做实际前向、反向和参数更新；prepare 会复用已下载的 1.7 GB FITS，正式数据准备成功后可以重复调用校验复用。smoke 在固定的 300 个开发对象上检查所有模型、特征和三折，不能打开正式测试集。all 顺序执行数据复用、正式 CV、全开发集重训、偏移模型训练、冻结、最终评估、重载验证。

脚本默认寻找同级目录 sdss-photoz。若位置不同，给每条命令补上参数：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage prepare -V1Root 'D:\AI\photoz-windows\sdss-photoz'
```

也可使用 -Python 指定现有 python.exe 的完整路径。所有命令都在 Windows PowerShell 执行，无需激活虚拟环境。

### 已知的间歇性 CUDA 崩溃

V2 不保证解决系统或 PyTorch 的访问冲突。模型试验在独立进程执行，kNN、随机森林、XGBoost 不会导入 Torch；其中任何已完整完成且文件校验通过的试验可以复用。失败立即停止并保留日志，不自动重试、不静默换设备。

如果选择 CPU 路线，用 -Device cpu 重新执行 check、smoke、all，输出默认写入独立的 runs/full_cpu。CUDA 路线默认 runs/full_cuda。只能在新测试尚未打开之前改变训练设备或配置。不要把不同环境的模型文件复制进另一运行目录冒充已完成任务。

更换终端后继续同一运行：进入本项目目录，重新执行相同的 all 命令即可。已完成的折或模型直接复用，尚未完成的那个试验从头训练，不是从中间 epoch 恢复。环境、代码、数据、线程数、设备、参数变化会触发身份校验，需要新的运行目录。

## 3. 实验设计与默认计算量

| 输入组 | 维度 | 特征 |
|---|---:|---|
| mag | 5 | 消光改正后的 u,g,r,i,z 星等 |
| colors | 4 | u-g,g-r,r-i,i-z |
| colors_i | 5 | 四颜色+i 星等 |
| colors_i_err | 10 | 四颜色+i 星等+五波段星等误差 |

全部特征不使用红移。标签是主光谱红移 Z，范围 0<Z<=7，保留 IS_QSO_FINAL==1 且至少三个有效消光改正波段的对象。误差字段是 PSFMAGERR，不是逆方差。SOURCE_Z、ZWARNING 只用于结果审计。

四颜色+i 星等在完整测光时可重构五星等，因此 mag 与 colors_i 比较表示方式；colors 与 colors_i 比较是否加入亮度；colors_i 与 colors_i_err 比较测量误差的增益。存在缺失值时，各表示分别填补可能使这种精确等价关系不再成立。

| 模型 | 候选配置 |
|---|---|
| kNN | k=1,5,15,31；等权/距离加权；k=1 两种权重等价，只保留等权，共7组 |
| 随机森林 | 200棵树、最大深度20、叶节点最少5或20条，共2组 |
| XGBoost | 最大1000轮、深度6或8、折内早停，共2组 |
| MLP | [128,64] 或 [256,128,64]，最大100 epoch、折内早停，共2组 |
| 常数 | 开发训练部分的均值、中位数 |

正式配置先在固定抽出的 50,000 个开发对象上做三折选模。各模型使用完全相同的外层折，kNN/MLP 的填补、标准化在每折训练数据内拟合；树模型仅填补。XGBoost/MLP 先在外层训练折内部留15%做早停，选定步数后重新初始化并用整个外层训练折重训，最后才计算外层验证指标。

每个模型×特征组按三折平均 MAE 选参数。最终在全部17万开发对象上重训，XGBoost/MLP 的训练步数取该配置三个折内最佳步数的中位数。CV 分数是开发选择指标，三折标准差不是置信区间，也不能代替新测试结果。

默认有 162 个 CV 折任务（包括6个常数基线任务）、18个最终模型、8个偏移模型。CV 只保留预测、曲线、指标和训练记录，不长期保存重复的模型文件；最终和偏移模型完整保存。候选数量并非严格等算力预算，报告需保留配置数量和耗时记录。正式耗时取决于CPU、GPU、磁盘和早停结果，不预填时间承诺。

在 configs/full.json 中可将 cv_max_samples 改为 null，在全部开发集上交叉验证；应在开始正式运行前决定，并使用新的运行目录。为保持对照一致，不单独给某个模型减少 CV 样本。

## 4. 亮度偏移实验

阈值预先固定为消光改正后的 i=20。对每个模型，用已由开发 CV 选出的 colors_i_err 配置训练两组：

1. bright：仅使用 i<20 的开发对象。
2. mixed：从开发集中所有 i 有效的对象中随机抽取。

两组训练样本数一致，最多50,000条；若亮源不足则共同缩小。两组采用相同超参数及固定训练步数，在同一批新测试集 i>=20 对象上评估。比较会改变亮度分布，也可能同时改变红移等分布，不能解释为只改变了一个纯粹的物理变量。

## 5. 输出与判读

| 文件或目录（相对运行目录） | 内容 |
|---|---|
| cv_folds.csv / cv_summary.csv | 每个外层折的结果、跨折均值和标准差、拟合/预测耗时 |
| cv_selection.json | 每个模型×特征组的入选配置、训练步数和主模型 |
| final_models.json / shift_models.json | 最终模型、偏移模型索引 |
| frozen.json | 在打开新测试前冻结的实验文件哈希 |
| trials/final_* | 最终模型、预处理器、元数据、训练曲线 |
| trials/shift_* | 亮度偏移模型 |
| logs/*.log | 每个试验独立日志 |
| report/metrics_test.csv | 新测试集上的全部18组结果 |
| report/feature_ablation.csv | 同一模型内的特征对照 |
| report/metrics_by_bin.csv | 红移与亮度分段指标及样本量 |
| report/shift_comparison.csv | 相同暗源对象上的亮源训练／混合训练对照 |
| report/cases.csv | 最多20个异常对象及最多20个正常对照、特征、预测和邻居摘要 |
| report/case_neighbors.csv | 来自开发集的最近15个颜色邻居及其光谱红移 |
| report/*.png | 对照图、预测图、折内训练曲线、颜色-红移图、异常对象邻居图 |
| report/RESULTS.md | 实际指标自动生成的报告 |
| report/reload_check.json | 所有最终模型和偏移模型的重载验证 |

异常样本按归一化残差绝对值排序，并预先按红移段分配名额；正常对照来自相同粗红移段且尽量匹配 i 星等。邻居搜索只在开发集的四颜色中进行，填补和标准化也只拟合开发集。邻居红移离散、测量误差大、缺失值等都是需要继续检查的线索，不自动当作物理原因或删样本理由。颜色-红移图显示开发样本的中位数和四分位范围，不能当作因果解释。

归一化残差 delta=(预测Z−真实Z)/(1+真实Z)，离群阈值 |delta|>0.15；NMAD=1.4826×median(|delta−median(delta)|)。不裁剪预测后报分，负预测数量单独记录。亮度分段统计只包含落在所列区间且 i 有效的对象。稀少高红移分段务必连同样本量解释。

所有模型在同一新测试集上比较。V1 和 V2 的测试对象不同，不能直接把 V2 与 V1 的指标差当成纯模型提升。本项目未验证跨巡天泛化。

## 6. 分步命令、预测及打包

需要分阶段运行时，可分别执行 train、evaluate、verify：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage train -Device cuda
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage evaluate -Device cuda
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage verify -Device cuda
Invoke-Item .\runs\full_cuda\report
```

evaluate 和 verify 的预测固定用 CPU；这里的 -Device cuda 用于定位 runs/full_cuda，并不再次在显卡上训练。

真实新对象推理 CSV 必须包含 object_id（可选）、psfmag_u...psfmag_z、extinction_u...extinction_z，误差 psfmagerr_u...psfmagerr_z 可缺失。输入星等尚未做消光改正，程序只减一次消光。

```powershell
$py = '..\sdss-photoz\.venv-win\Scripts\python.exe'
& $py .\v2.py predict --run runs/full_cuda --model primary --input ..\sdss-photoz\data\processed\example_inputs.csv --output predictions_new.csv
```

完整实验打包（包含模型与 V2 数据，不包含原始1.7GB FITS或虚拟环境）：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage bundle -Device cuda
```

生成 sdss-photoz-v2-results.zip，支持 Zip64。已有同名压缩包时不会覆盖，先移动旧包。恢复该结果包后，运行目录统一为 runs/exported；在依赖匹配的环境中可用 v2.py verify --data data/v2 --run runs/exported 验证。正式运行目录建议单独备份，Git 仓库只提交代码与配置，数据和训练产物按 .gitignore 排除。

## 7. 验证与边界

```powershell
$py = '..\sdss-photoz\.venv-win\Scripts\python.exe'
& $py -m unittest discover -s tests -v
```

测试覆盖身份去重桥接、旧样本排除、折内/折外分离、训练集预处理、特征不读标签、匹配样本量的偏移设计、指标及测试集冻结。validation/ 中记录交付前的实际技术验证范围。小规模真实目录片段仅用于技术检查，其结果标记 TECHNICAL_CHECK_ONLY，不能用于项目正式性能或简历。

本包是可运行源代码，不包含预训练的 V2 正式模型。你的 Windows CUDA 环境需通过 check 和 smoke；此前的间歇性访问冲突不能由代码包宣称已经修复。

## 参考资料

- SDSS DR16Q目录：https://www.sdss4.org/dr17/algorithms/qso_catalog/
- astroML kNN测光红移示例：https://www.astroml.org/examples/learning/plot_neighbors_photoz.html
- scikit-learn交叉验证：https://scikit-learn.org/1.6/modules/cross_validation.html
- scikit-learn预处理与泄漏：https://scikit-learn.org/1.6/common_pitfalls.html
- XGBoost 3.0 API：https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html

