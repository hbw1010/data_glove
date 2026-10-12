# 数据格式与语义

本页是输入、标定和结果格式的唯一说明。操作步骤见 `workflow.md`。

## 1. 原始录制

统一内部表示为 `Recording`：

| 字段 | 类型/单位 | 约束 |
|---|---|---|
| hand | left/right | 普通录制取录制身份；新版实时录制取本次明确的主机手侧选择 |
| device_id | 十六进制字符串 | 小写，去掉可选0x前缀，不丢失前导零 |
| cs_by_joint | 21个整数 | 原始J1..J21对应物理CS，须为0..20的不重复排列；不随用户手侧选择改写 |
| timestamps_ns | N个int64，ns | 非空且严格递增，不转成浮点绝对时间 |
| angles_deg | N×21，degree | 有限的原始绝对角[0,360) |
| sequence / drops | 可选整数序列 | 以来源元数据区分DGST消息号和兼容模式主机交付号；不能冒充物理扫描号或生成零drop证明无丢扫 |

当前读取器针对明确结构支持：

- SD HDF5：`dataglove.h5` 内的 `/sensor/encoder/timestamp_ns` 和 `joint_deg`，身份/标定从内嵌JSON与metadata读取。
- 联合录制：`sensors.h5` 加同目录 `manifest.json`，编码器组保存板端时间与21路角，设备元数据中包含角度约定与录制时标定。
- 短采集 JSON：包含 frames，以及手侧、设备、通道、角度语义等元数据。
- 实时录制：DGST USB或ADB兼容模式的`encoders.jsonl`，加同目录`device.json`、`sensors.json`、`encoder_calibration.json`、`host_session.json`等元数据。
- compact-v2：`data/encoder.csv`，按明确的时间和J通道列解析，同时读取同会话元数据。
- pico原始采集：设备目录内的 `dataglove/sensors.jsonl`，配套 `capture.json`、`association.json`、`metadata.tar`。只读验证录制完成状态、归档/编码器文件哈希、设备/手侧/boot/stream/录制UUID和帧数；归档必须具有完整且不重复的固定元数据成员。直接解析原帧，不再要求先复制改名。若旁边存在 `stereo-camera.h5`，须与录制关联证据一致才记录 `stereo_path`。
- 双手采集 HDF5：`v21_glove_capture` schema v1，通常命名 `*.v21glove.h5`。`/left`、`/right` 分别包含设备身份、`metadata_tar`、`encoder` 和可选 `video`；按内嵌归档校验手侧、设备、接线与已应用标定，恢复原始绝对角后沿用同一数值链。读取器单侧调用必须指定 `hand=left/right`，不能借此覆盖源身份。
- 双手对齐 HDF5：`v21_aligned_60hz` schema v1，通常命名 `*.aligned_60hz.h5`。必须经独立手动导入接口提供左右手完整项目标定及各自角度约定，不能经普通 metadata 补全接口伪造采集证据。

这不代表可以读取任意CSV或任意HDF5。目录有多个候选录制文件时应指定明确文件，不自动猜选。

### 双手采集 HDF5 的来源与时间

会话头必须提供有效完成时间等字段。文件没有原生 `recording_uuid`，因此以 `schema/schema_version/host/created_realtime_ns/start_host_mono_ns/main_output_path/part/sequence_number` 的排序紧凑 UTF-8 JSON 做 SHA256，前缀为 `derived-v21-glove-capture-v1:`；元数据明确标注 `recording_uuid_source=derived_session_header`。复制或搬移文件不改变该身份；联合导出仍要求两侧来自同一实际文件。

编码器统一时间轴取 `encoder/host_mono_ns`，不是 `host_recv_mono_ns`；采用文件中录制器已映射的主机单调时钟，不再次估计或重采样。`board_ts_ns` 及接收时间保留摘要，`boot_ids` 和采集 `errors` 原值保留。多个 boot 缺少逐帧归属时明确警告，不能声称同一启动周期；`errors` 不是丢帧数，不补造 drop 数据。

已声明 `calibrated_signed_pm180` 的角度通过录制时归档标定反算一次原始角。同一份零位和方向可作为内存微调基线；它不是新采集的零位证据。没有本地匹配标定时界面自动选择内嵌基线，不自动写入设备标定文件。后续微调与现有原始角路径共用，避免重复减零位。

### 双手对齐 HDF5 的手动参数与时间

`/time/unix_ns` 是两侧共同目标时间轴，必须严格递增且与头部帧数、首末时间一致；`/time/relative_s` 必须与其相对时间一致。声明频率固定为 60Hz，检查逐步整纳秒间隔，允许每步纳秒量化产生整段累计舍入，不用重新生成的 `arange(N)/60` 替换源时间轴。

`/glove/{left,right}/encoder/angles_deg` 是 `(N,21)` 有限角度数组；角度约定由用户逐侧明确提供。已校准有符号角根据上传的采集零位/方向反算一次原始角；原始绝对角保持原值。`source_frame` 是采样引用，允许重复，不当作真实 sequence；`source_time_delta_ms` 仅保存对齐误差摘要，不补造 drop。

完整标定深拷贝保存在 `manual_calibration`；`identity_source`、`cs_mapping_source`、`angle_convention_source`、`calibration_source` 均为 `user_supplied`。读取不访问头部另一台机器的原始文件路径。整个对齐 HDF5 的 SHA256 作为内容身份，`recording_uuid` 为 `derived-v21-aligned-60hz-v1:<sha256>`，并检查读文件期间来源未变。

同目录 `<session>.aligned_60hz.left.mp4` / `.right.mp4` 直接服务，不转码、不复制至源目录。核对 `<session>.alignment_report.json` 的来源 basename、双侧视频声明、帧数和首末时间；核对 `<session>.alignment.csv` 的逐帧目标时间、本侧编码器采样引用与时间差；核对 MP4 的实际帧数与 60fps PTS。允许合法 B 帧。源 HDF5、报告、CSV或视频变化后要求重新打开。视频时间保留真实 PTS，回放时间对应 aligned Unix 目标帧，不宣称原始曝光级硬同步。

### 标准关节 J 与原始采集列

原始帧的“J1..J21”只是该次采流的列名；左右共用的标准关节定义唯一维护在 `src/glove_calibrate/joint_channels.py` 的 `_JOINTS`，不从旧左/右机械配置推断。两侧机械JSON只保存对应具名关节的方向、比例与参数，不重复保存source_channel。

| 标准J | 从拇指到小拇指的关节含义 | 拇指对应URDF |
|---|---|---|
| J1～J5 | 侧摆 | J1=`thumb_cmc_abd_joint` |
| J6～J10 | 掌骨屈伸 | J6=`thumb_cmc_flex_joint` |
| J11～J15 | 次末节屈伸；拇指MCP、其余PIP | J11=`thumb_mcp_joint` |
| J16～J20 | 末端屈伸；拇指IP、其余DIP | J16=`thumb_ip_joint` |
| J21 | 小拇指CMC | `pinky_cmc_joint` |

这一定义同时约束显示和真正的模型输入，不只是改名。旧配置中J1驱动拇指屈伸、J6驱动拇指侧摆的关系已纠正；原始计数/角数组和两侧CS表不因此互换。旧标定仍按原始列存储，但在错误关节对应下确认过的拇指方向应重新检查；历史处理文件不会自动更改，重新处理才采用新定义。

左右物理CS映射分别维护在 `src/glove_calibrate/joint_channels.py` 的 `_CS_BY_JOINT["left"]` 与 `["right"]`，不互相镜像推算。修改左手映射不会改右手表；两侧仍各自保留机械参数、direction、scale、模型和设备零位。机械JSON通过 `engine.profile(hand)` 装配为完整解析配置，不再各自保存重复CS表。

当前接线以用户提供的 `a9ea11ef816df166` 左手图及图中右手参考列为底表；左手拇指两路按后续实物反馈修正：侧摆J1取CS15，CMC屈伸J6取CS7。只交换左手物理输入，不交换标准J语义或右手映射。表内数字均为CS，不是原始数组下标：

| 标准J | 左手CS（依次） | 右手CS（依次） |
|---|---|---|
| J1～J5 | 15、8、9、10、11 | 12、11、10、9、8 |
| J6～J10 | 7、20、17、1、6 | 13、18、0、3、4 |
| J11～J15 | 14、19、16、2、5 | 15、19、16、2、5 |
| J16～J20 | 13、18、0、3、4 | 14、20、17、1、6 |
| J21 | 12 | 7 |

除上述左手拇指修正外，其余通道仍按接线图；图中仅部分关节有实物调试记录，不代表全部通道已验收。旧左手标定若携带不同的`capture_cs_by_joint`会被拒绝，不自动改写或覆盖；应按新接线重新核对并采集。右手参考表与此前右手表一致。

标准J通过物理CS定位原始列：`source_index = raw_cs_by_joint.index(cs_by_joint(hand)[J - 1])`。实时接口的 `joint_layout` 按标准J排列，逐项提供 `channel`、`joint_name`、中文`label`、`cs`、`source_index`；码表读取 `frame.angles_deg[source_index]`。原始角数组、出厂接线、文件列序和采样证据不因此重排或覆盖。映射不可用时不猜列号。

标定表也按标准J显示，但零位和方向仍按原始列键读写。例如标准J1映射到原始第14列时，应读写 `zero_offsets_deg.J14` 与 `encoder_directions.J14`，不是同名J1。既有标定文件格式保持不变。

### 实时采集输出

动作录制与各组实时标定窗口写入 `data/raw/` 下的唯一目录，正常完成后发布 `encoders.jsonl`，配套保存本次身份、传感器、标定状态元数据。DGST帧的角度保留线上实际语义；读入时按下文统一恢复原始绝对角，而不是把板端标定角误标为原始角。ADB兼容模式角度直接由真实原始计数换算，区别见下节。

手侧有两份独立证据：设备原始声明保留在原样保存的元数据中；`host_session.json` 记录 `hand_selection_version=1`、`hand_selection=explicit_user`、用户选择的 `hand` 和设备声明 `device_hand`（未知为null）。JSONL帧保留设备手侧身份；读取器先严格核对设备/帧事实，再仅对带该版本标记的主机会话采用用户手侧。旧无版本录制沿用原事实解析，不追溯改写。普通离线补充元数据仍不能覆盖手侧。

用户手侧决定模型与解剖CS表，不改变原始21路角度或CS顺序。若板端已扣零，恢复原始角仍核对板端声明及原始接线并使用录制时的板端标定；不能拿用户所选手侧冒充板端标定身份。六组标定通过原始CS到所选手侧解剖CS的对应关系取样。

动作录制保留开始时所选本地标定的 `host_calibration.json` 副本。接收中的文件使用 `.partial`，只有正常停止且通过完整性检查才成为可导入录制；中断时保留证据和错误信息。设备序号证明DGST消息连续性，不等于证明板端所有物理扫描都被交付；没有真实drop字段时不生成全零drop数组。

HTTP实时状态里的绝对纳秒时间和序号同样使用十进制字符串；`hand` 为用户选择，`device_hand` 为设备原始声明。当前角度统一为原始绝对角，状态同时提供帧龄；过期样本不允许用于新标定。

### 无相机ADB编码器兼容模式

- `transport=adb-ma782-v1`，只读板端已有`/usr/bin/ma782-read`和`/dev/ma782-array`；不启动相机、DGST或写入板端。
- `device.json`、`sensors.json`、`encoder_calibration.json`分别使用`dataglove.host_encoder_device`、`dataglove.host_encoder_sensors`、`dataglove.host_encoder_calibration`版本1。三文件tar在主机生成并计算SHA256，不冒充固件返回的DGST `metadata.tar`。
- `metadata_source=host_generated`、`identity_source=host_generated`。`board_device_id`是真实SoC序列号，小写、去可选`0x`前缀但保留前导零；`device_id=UUID5(NAMESPACE_URL, "dataglove:board-soc:" + board_device_id).hex`。后者绑定本地标定，独立于DGST身份；不得直接复用另一ID的零位。
- `mapping_source=board_factory`时，出厂区头、长度、SHA256、schema、设备ID和21路排列均经校验，原出厂JSON保留于`factory_record`。只有确认无出厂记录才用显式手侧配置，标为`explicit_hand_profile`、`mapping_verified=false`；损坏或冲突拒绝。
- 21路角度为`raw_count * 360 / 65536`，按真实CS映射排列，完整保留计数精度，不使用工具四位小数显示值。`angle_convention=raw_absolute_0_360`、`calibration_applied_to_samples=false`；板端标定状态为`not_read`，不据此声称板端未标定。
- `timestamp_ns`使用板端`CLOCK_MONOTONIC`物理扫描中点：`snapshot_timestamp_ns - scan_duration_ns // 2`。只在同一`boot_id`内比较；启动与采集中检查重启，变化后终止会话。
- 顶层`sequence`是从1开始的主机交付计数。每帧`source_sample`独立保留驱动`scan_sequence`、扫描结束时间、时长、周期、CS0..20整数计数、installed/transfer/suspect掩码、overruns、扫描/逐CS错误计数和`boot_id`。同一扫描去重；逆序、同号异数据、计数器复位、SPI失败、截断、超时或缓存溢出显式终止。
- 读取的是驱动最新快照，不保证交付所有物理扫描。物理跳号保留在`source_sample.scan_sequence`；不得把主机连续编号当作无丢扫证据。Suspect rail保留，不擅自填零或滤除真实极值。
- 不采集视频。停止兼容连接仅确认主机ADB进程已终止，不伪造DGST STOP/END。停止动作录制仍保持实时读取。

### 角度归一化

输入元数据必须明确原始角语义。`raw_absolute_0_360`（或格式中明确等价标志）直接进入计算。`calibrated_signed_pm180` 必须有录制时对应的完整板端标定，按每路公式恢复：

```text
raw = (board_zero + board_direction * recorded) mod 360
```

新标定只在恢复原始角后应用一次。不能从值域猜角度语义；不能用当前设备的另一份零位替代录制时板端零位。

### 缺失元数据补全

只允许明确补齐：`hand`、`device_id`、`cs_by_joint`、`angle_convention`、`angle_unit`、`channels`、`calibration_applied_to_samples`。界面接受JSON对象，命令行接受 `--metadata` JSON文件。

例如字段形式如下，其中接线必须来自实际录制，不可照抄另一只手套：

```json
{
  "hand": "right",
  "device_id": "实际十六进制设备ID",
  "angle_convention": "raw_absolute_0_360",
  "angle_unit": "degree",
  "channels": 21,
  "calibration_applied_to_samples": false
}
```

该示意故意不提供虚构的CS表或可用设备ID。已知事实冲突会拒绝导入；已应用板端标定不能由此接口补造。

## 2. 本地标定

固定文件名为 `left_<device_id>.json` / `right_<device_id>.json`，保存在项目 `config/dataglove/v21/zero/`。

沿用 `UrdfZeroProfile` 版本1：

- `hand`、`device_id`、`channels=21`、`angle_unit=degree`。
- `stream_zeroed=false`、`stream_range_deg=[0,360]`。
- `expected_cs_by_joint`：标定所绑定的原始通道接线。
- `capture_cs_by_joint`：按所选手侧独立映射、以统一解剖J定义进行分组采样的接线。
- `zero_offsets_deg`：原始列J1..J21各自的零位角，正式文件必须完整；不是标准J的直接索引。
- `encoder_directions`：原始列J1..J21每路必须为整数+1或-1，显示及编辑需先定位原始列。
- `captured_groups`：thumb、index、middle、ring、pinky、pinky_cmc六组。
- 额外来源信息保留所选录制与窗口，不改变计算字段语义。

草稿可以包含未完成的null零位，但不得以正式标定保存或用于全段处理。预览会在非线性机构换算前冻结尚未标定的通道。

标定静止窗口推荐1.5秒，至少100帧且首末样本跨度至少1秒；中央80%环形行程不超过1.5度。存在序号/drop时按其声明语义检查交付连续性，没有时仅检查时间间隔；无论DGST消息号还是兼容模式交付号连续，都不能证明所有物理扫描无丢失。

### 录制级零位修正快照

`data/corrections/correction_<uuid>.json` 独立于全局设备标定，`schema=glove_v21_recording_correction`、`schema_version=1`。保存完整 `base_calibration`、按标准J1..J21排列的 `zero_adjustments_deg` 和 `encoder_directions`、`source`来源身份与内容SHA-256、`source_path`及创建时间。快照每次生成新文件，不覆盖原始录制。显式保存同时合成全局有效零位与方向，原子替换对应 `config/dataglove/v21/zero/<hand>_<device_id>.json`，移除录制专属 `recording_correction`，不把分帧局部增量写入通用标定。直接导出的自动快照保存不替换原标定。

来源绑定原文件SHA-256和规范化后的角度、时间戳、序号、drop、手侧、设备、原始CS及帧数；路径不参与匹配，相同内容搬迁后仍可加载。来源指纹每个打开的录制计算一次，不在拖动预览时反复散列整段。

录制打开描述提供来源严格匹配的最新有效 `saved_correction`，由快照内带时区的 `created_at` 倒序确定，不按UUID文件名排序；无匹配则为null。界面用该快照恢复完整微调状态，原始采集标定与目标修正分开。保存后的磁盘全局零位不能再与原增量叠加。

增量是编码器零位角的变化，单位度，不是URDF输出关节角变化。通过标准J→该侧物理CS→基础标定原始列定位，`effective_zero=(base_zero+delta) mod 360`。修正快照的 `encoder_directions` 是完整标准J的目标绝对方向，只允许整数 `+1/-1`，同样按CS映射回原始标定通道；不是反向次数或方向增量。旧快照缺少该字段时从冻结的基础标定恢复方向，不默认全部正向；显式损坏字段拒绝。方向作用于本手全部帧，局部区间仍只调整零位；归零不复位方向。比例、原始角及采集时已应用的标定不变，不用新方向再次反算源角度。基础标定的原始列接线仍必须与录制一致。

即时预览和正式导出共用相同的有效标定计算。正式HDF5的 `calibration_json.recording_correction` 内嵌修正名称、来源、完整基础标定和增量；即使之后修改全局标定或网页草稿，也不影响已经提交或导出的快照。预览仅计算当前帧机构/URDF，正式导出才运行全部三阶段。

`frame_adjustments` 是本侧帧号的已排序、无重叠数组：`[{start,stop,zero_adjustments_deg:{J1:..., ...,J21:...}}]`，采用半开区间 `[start,stop)`；旧快照缺该字段视为空。局部增量叠加在已合成的全局零位上，只在区间内生效，区间结束恢复全局。界面重叠编辑先切分并覆盖旧区间，再保存规范化列表。导出预验证全部区间和来源，在段边界切换有效零位，不重建模型或丢弃IK历史，不修改原始角度和时间戳。

经明确授权对失效编码器做估计补全时，可用逐帧局部零位增量表达等价的角度补偿，原始文件及可信实测帧保持不变。基础标定的`encoder_repair`附加信息记录估计通道、来源指纹、区间、方法、验证误差和独立逐帧报告路径，并随`calibration_json`保存。此类结果包含估计值，不能作为真实测量恢复；零位微调本身也不能恢复长时间缺失的运动。运行时标定计划不为每帧重复复制整份来源快照，完整区间表仍保存在导出元数据中。

## 3. 三阶段结果HDF5

单手格式根属性：`schema=glove_v21_processed`、`schema_version=1`、`hand`、`device_id`、`complete`、`valid_counts`。联合格式根为 `schema=glove_v21_pair_processed`、`schema_version=1`、`recording_uuid`、`complete`；下表所有路径分别置于 `/left`、`/right` 组中，每组带完整单手属性，两手长度可不同。

| 路径 | 形状/类型 | 含义 |
|---|---|---|
| `/timestamp_ns` | N，int64 | 原始时间戳 |
| `/source_sequence`、`/source_drops` | N，可选整数 | 仅原记录真实存在时保存 |
| `/exoskeleton/joint_names` | 21，UTF-8 | V21 URDF具名关节顺序 |
| `/exoskeleton/qpos_rad` | N×21，float64 | 机构换算后的外骨骼关节弧度 |
| `/hand/keypoint_names` | 21，UTF-8 | MediaPipe骨架关键点顺序 |
| `/hand/keypoints_m` | N×21×3，float64 | 拟合后的人手关键点，米 |
| `/wujihand2/joint_names` | 20，UTF-8 | 实际求解器返回的DOF顺序 |
| `/wujihand2/qpos_rad` | N×20，float64 | Wuji Hand2关节弧度 |
| `/<阶段>/valid` | N，bool | 本帧该阶段是否有效 |
| `/<阶段>/error` | N，UTF-8 | 失败原因；有效帧为空 |
| `/metadata/calibration_json` | UTF-8 JSON标量 | 使用的完整标定快照 |
| `/metadata/recording_json` | UTF-8 JSON标量 | 输入身份、路径、录制元数据 |
| `/metadata/processing_json` | UTF-8 JSON标量 | 坐标、单位、算法配置和资产指纹、依赖版本 |

同一索引始终对应同一个输入采样时刻。阶段无效时数值为NaN，浏览器JSON接口则为null；不能将NaN当角度传给模型。上游失败使依赖它的下游无效，下游失败不抹掉上游结果。

### 坐标系与拓扑

人手关键点顺序：腕部，拇指4点，食指4点，中指4点，无名指4点，小拇指4点。手型为固定参数化骨架拟合，不是标准MANO网格或个人beta估计。坐标为对应V21模型的局部根坐标，米；左手规范化过程会镜像观测进行拟合再镜像回对应侧。精确描述保存在 `processing_json.hand_coordinate_frame`。

Wuji重定向在内部另有腕部坐标转换，不能把输出骨架坐标直接当作Wuji模型世界位置。URDF关节角按各模型原始轴定义，不通过比较列号判断左右手同一关节；使用名称。

关键点编号与[MediaPipe HandLandmark](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/hands.py)一致：0为腕，1–4为拇指CMC/MCP/IP/指尖，5–8为食指MCP/PIP/DIP/指尖，9–12为中指，13–16为无名指，17–20为小指。网页显示全部21点，并使用官方[HAND_CONNECTIONS](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/hands_connections.py)连接手掌和手指。

为去掉拇指掌骨附近的折返，拟合后按`P[1] = (P[0] + P[2]) / 2`重建CMC点，其余20点不变。此点是几何估计，不是MediaPipe检测或真实CMC位置测量。显示、数值面板、`keypoints_m`导出及Hand2重定向使用同一套重建后的21点；不隐藏CMC，也不画圆滑曲线。

`processing_json.hand_keypoint_postprocess`记录`order=mediapipe_21`和CMC的`index=1, method=midpoint, reference_indices=[0,2]`。对齐的是编号与连接关系；坐标仍为本工程米制URDF局部坐标，并非MediaPipe图像归一化坐标或以手部几何中心为原点的world输出。旧文件不改写，回放忠实展示其原有坐标；应用新CMC规则须重新处理。

### Hand2自碰撞约束

`processing_json.wujihand2_self_collision`保存本次求解的实际策略。Web单手或双手导出通过严格布尔 `self_collision` 逐任务选择，默认关闭，写入`{"enabled":false}`；选项在提交时冻结，两侧共用，不修改YAML配置或已排队任务。仅显式启用时才记录碰撞几何、活动对、机械装配排除对、间隙与最终校验策略；不改变外骨骼和人手两个上游阶段。`/api/results/open`以同名字段返回；历史文件缺少该字段时返回null，不能将未启用或未知状态宣称为已经防碰撞。

启用碰撞优化时，左右Hand2各使用26个完整collision STL、295个活动对。除同刚体及单活动关节直接装配邻接外均检查；5个掌部二轴装配对使用原始STL表面距离，避免凸包填平真实装配凹槽，其余使用保守凸包。启用时的默认最小间隙0.5mm；左手掌部与拇指`proximal_abd`的CAD装配最大间隙约0.116mm，配置明确仅将该对设为0.05mm。数值容差为0.0001mm。

启用时，SLSQP求解使用距离硬不等式约束；提交历史状态前及滤波后再次独立校验。失败仅令该帧Wuji阶段无效，不以旧姿态冒充成功，也不修改有效的外骨骼和人手数据。关闭时不构造自碰撞检查器，不添加或校验这些约束，可能出现穿模。此检查不包含排除的机械邻接对、外部障碍或采样间连续运动，不能作为真机安全认证。

### 时间与顺序

绝对纳秒时间在HTTP中以十进制字符串传输，避免JavaScript丢失64位整数精度。`time_s`为相对首帧的浮点秒，仅用于播放。处理默认不额外低通滤波，但骨架拟合和Wuji求解仍保留上一帧warm-start/时序约束，因此必须顺序处理，不能以预览跳转顺序导出。

### 模型一致性

结果保存项目内模型资产SHA-256。回放逐项校验这些资产，不匹配时拒绝显示，防止用更新后模型解释旧轨迹。数字结果仍可用通用HDF5工具读取；兼容历史模型应保留其对应项目资产版本。

## 4. 可选腕相机回放关联

外置双目腕相机关联接口仍可单独读取，但当前微调和结果页面均不展示它。该接口优先读取所选侧 `metadata/recording_json.metadata.stereo_path`，也支持以下旧关联格式；它与 dataglove 自身相机是不同来源。

例如 `data/processed/pico_demo2_left_a9ea11ef816df166.h5` 的关联文件为同目录 `pico_demo2_left_a9ea11ef816df166.replay.json`：

```json
{
  "schema_version": 1,
  "stereo_path": "/home/boen/Desktop/pico_demo2/left-stereo-camera-000002/stereo-camera.h5"
}
```

相机HDF5必须是完整的 `stereo_camera_dataset` v2。后端校验相机手侧及其 `dataglove.association_json`，根据其中的SHA-256核验原目录 `dataglove/metadata.tar`，再从归档 `device.json` 核对手套 hand/device_id/boot_id。相机的 `metadata.device_id=rv1126b` 不是手套ID。

`/sensor/camera/cam0`、`cam1` 的 `timestamp_ns` 必须严格递增，`frame_index` 为逐帧 `0..N-1`，`video_path` 必须位于该HDF5父目录内。首次加载使用 FFmpeg 的 `ffprobe` 读取真实视频PTS并核对帧数；媒体接口支持HTTP Range，关联文件变化后要求重新打开。

API相机 `time_s` 以两路最早首帧为共同零点，保留相位差；`video_time_s` 是真实逐帧视频PTS；`timestamp_ns` 为原始纳秒十进制字符串。腕相机的共同零点与手套首帧不是同一物理时刻；未提供跨时钟映射时不能宣称严格同步。相对联动不改原始时间戳、不重采样、不回写源文件。

## 5. Data glove 与头部相机回放

Pico手套自身相机由同一 `dataglove/` 下的 `video.h264`、`camera.jsonl`、`capture.json`、`association.json` 和 `metadata.tar` 关联。`camera.jsonl` 的每条视频记录包含 `sequence`、`timestamp_ns`、`stream_id`、`keyframe`、`offset`、`length`；按关联清单核验文件大小和SHA256，并检查流身份、帧数、严格递增时间戳以及视频字节区间。它不是旁边 `stereo-camera.h5` 的外置双目。

浏览器不能直接定位裸H264。后端在项目 `.cache/glove-video/` 内容寻址缓存中无损封装MP4，再以FFprobe核验解码帧数、尺寸和逐帧PTS；不重编码、不写原始目录。视频API支持HTTP Range。`video_time_s`为封装后实际PTS，`time_s=(camera_timestamp_ns - encoder_first_timestamp_ns)/1e9`，二者通过帧索引映射；绝对纳秒仍用十进制字符串传输。Pico手套相机和编码器共用同一boot/stream的单调时钟，但不据此承诺曝光级硬同步。

双手采集 HDF5 的视频读取同手组 `video/h264` 逐帧变长 uint8 Annex B 数据及 `host_mono_ns`、`sequence`、`keyframe` 等证据。逐帧提取至临时缓存，核对包边界、关键帧、帧数、分辨率及帧率后复用无损封装流程；两侧缓存隔离。相机与编码器均使用本文件主机映射时间轴，不假装同 boot。无 `video` 组明确为缺失，有组但损坏则阻止准备。只关联文件内另一手视频，不根据相邻文件名猜测头部或其他相机。

结果页通过本侧 `recording_json.path` 关联同一组源文件，并额外核对所选结果的手侧、设备、录制身份、帧数及每个编码器纳秒时间戳。联合文件分别打开左右两侧结果句柄，视频接口使用各自`/api/results/{id}/glove-video`命名空间；不能拿另一侧或外置相机替代。模型和视频共用公共相对秒轴，视频仍保留上述真实偏移。

逐帧探测使用4个解码线程；同一服务进程内按解析后的文件路径、设备/inode、大小、mtime/ctime及预期帧数/尺寸缓存成功结果，文件变化会失效并重新探测，探测前后也核对身份。缓存不跳过来源SHA256和关联校验，不把旧PTS套用到变化后的文件；服务重启后需重新探测。

例如Downloads/pico_demo2左手视频1011帧，首帧比编码器晚0.154078063秒；右手1020帧，首帧比编码器早0.029401152秒。这些偏移必须保留，不能把两者首帧强行设为同一时刻。

其余相机元数据通过原始目录中相同 `recording_uuid` 发现：另一手套自身视频和 `head_*/*.h5` 中的 `head_dataset` v1相机，不聚合外置腕部双目。另一手套视频以其编码器首帧为原点；头部RGB/IR保持原设备共同起点，并核验帧索引、时间戳、路径和解码帧数。微调界面仅创建双glove与RGB_left、RGB_right共4路播放器，采用加宽两列布局，不请求IR视频；隐藏IR不改变RGB时间偏移。四路共用公共时间轴，结果页仍只显示双glove，不承诺跨设备硬同步。

## 6. 双手联合处理

双手导入必须来自唯一左、右手候选、不同设备及同一非空 `recording_uuid`。两侧分别保留完整原始采样率、首帧时间、帧数和序号；公共轴只是显示上的相对时间，不重采样、不合并原始帧索引。若公共时间超出较短一侧范围，该侧明确无帧。

两侧标定、全局增量、局部帧段、来源指纹及修正文件独立保存。双手提交全部验证后冻结两侧参数，在各自原始帧顺序中运行共享三阶段求解器。目录录制只在任务根目录发布一个 `processed.h5`；单文件采集在源文件同目录发布 `<源文件stem>.processed.h5`，例如 `session.v21glove.processed.h5`。`/left` 和 `/right` 各自内嵌修正和真实时间轴，不伪造帧对齐。只有两手完整写完才原子发布；失败或取消保留旧成品。仅允许替换相同联合schema、版本和 `recording_uuid` 的旧文件，拒绝覆盖其他数据或符号链接。
