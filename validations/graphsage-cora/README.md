# Native Cora GraphSAGE appliance adapter

`../cora_device` が準備したケースを、独立した
`validations/graphsage-cora` パッケージへ書き出す。対象は全近傍 mean
GraphSAGE の推論である。第 1 層は自己・近傍別の重み、bias、ReLU、L2
正規化を使い、第 2 層は正規化しない logits を返す。暗黙の自己辺は加えない。
層間のホスト呼び出しはなく、`f_run` を一度呼ぶ。

`appliance.py`、`appliance_worker.py`、`portable.py` の転送・回収手順と
`sdk_runner.py` の実行記録は、
`../../hierarchical_gnn/grouped_csr/native_cora_appliance/` の E217 実装から
移した。既存の `SdkCompiler` / `SdkLauncher` 経路を使い、各 appliance
step は compile または run を一回だけ行う。監査は回収後の host step
で実行する。書き出し自体はジョブを提出しない。

エクスポーターは元の `case.json`、`input_model.npz`、`arrays.npz`、
`node_ids.npy` のバイト列を保存する。`arrays.npz` だけは保存名を
`inputs.npz` に変える。ノード配置の NPZ コピーは元 NPY と照合する。
CSL ソース、入力、学習記録、重み、アダプター、生成物、実行要求、回収物の
ハッシュを結び付ける。Cora ケースには `training.json` と `weights.npz`
が必要であり、学習記録の入力・重みハッシュ、元コーパスのハッシュ、学習・検証
split を確認する。元コーパスのバイト列は同梱しない。監査の出発点は、学習記録と
結び付いた元の特徴量・辺出現列・パラメーターである。

`saved_audit.py` はデバイスの packer を import しない。元の辺出現列から
特徴量の平均を FP64 で計算してから射影し、全ノードの preactivation、hidden、
logits を照合する。別のデコーダーが特徴量の列分割、両重みバンク、複製
パラメーター、重複辺の個数と同一 source-group 内の順序、近傍平均の係数、
ノード配置を検査する。出力は `rtol=2e-4, atol=2e-5`、argmax は全ノード一致、
padding は正のゼロのビット列、完了・演算・collective カウンターは完全一致を
要求する。ゼロ近傍とゼロノルムも同じ契約に含む。

転送量を抑えるため、読み戻す入力は最後にアップロードした `slots` だけで
ある。その witness と初期カウンター、完了後のカウンター・時計・全出力を
回収する。重みや特徴量全体は読み戻さないため、保存入力の同一性と `slots`
witness は全入力のデバイス上での不変性を証明しない。JSON のキー順には依存せず、
明示した順序でアップロードする。

宣言済み状態は PE あたり 44 KiB 以下を入口条件とし、run 前に生成された ELF の
低位 SRAM 配置終端が 44 KiB 未満であることと、全 RPC symbol・`f_run` ABI を
確認する。コンパイラー本体は解析しない。動的 stack の最大使用量はこの検査に
含まれない。ELF の実バイト列を検査するのは launch 前の worker である。保存する
capture には ELF 自体ではなく、そのハッシュと配置の inventory を含める。
オフライン監査はこの記録の同一性を確認し、ELF 自体を再ハッシュしない。
各ケースは 128 MiB、回収物は 256 MiB を上限とし、archive の展開量、
NPZ の展開量、ファイル数、パス、ELF の走査時間にも制限がある。

想定ケースは synthetic tiny、Cora の 64 ノード誘導部分グラフ、全 Cora の
16×16、16×32、16×64 PE である。native ケースは元の 2708×1433 入力と 1433→16→7
重みを保持する。64 ノードの近傍平均は誘導部分グラフで再計算するため、全 Cora
の精度を代用しない。学習・検証 split の正解数を報告するのは全 Cora ケースだけ
であり、未使用の test split は評価しない。時計は同じ PE の開始・終了差分であり、
同期した全体実行時間ではない。

ローカルでの準備例は次のとおり。通常の `tools/run_with_budget.py` の資源枠内で
実行し、ケースは既存の device driver で事前に準備する。

```text
python3 -B -m unittest discover -s models/graphsage/cora_appliance -p 'test_*.py' -v
python3 -B models/graphsage/cora_appliance/export_appliance.py \
  --case tiny=research/runs/E219/tiny-2x2 \
  --case native64=research/runs/E219/native64 \
  --case full-c16r16=research/runs/E219/full-c16r16 \
  --case full-c16r32=research/runs/E219/full-c16r32 \
  --case full-c16r64=research/runs/E219/full-c16r64 \
  --training research/experiments/data/E219/training.json \
  --weights research/experiments/data/E219/weights.npz \
  --output research/runs/E219/export-graphsage-cora \
  --request-id graphsage-cora-e219 --compile-workers 1 --host-timeout 600
```

出力を sibling validation repository の `validations/graphsage-cora` に置き、
登録ファイル `validation.json` と SDK 不要テストを検証した上で、生成された
`request-template.json` を既存の要求登録手順に渡す。各ケースの順序は
compile → run → audit である。compile は最大 1200 秒、worker は既定 600 秒、
host audit は最大 120 秒である。これらは要求の上限であり測定値ではない。
ケースを増やす際は先行ケースの receipt から予測し直す。

回収済み `capture-data` だけから再監査できる。

```text
python3 -B validations/graphsage-cora/saved_audit.py /ABS/capture-data --out /ABS/fresh-audit.json
```

SDK 不要テストは mock runtime による正常回収、各出力・入力 witness・カウンター・
carrier の破損、重み分割・辺出現の破損、失敗時の stop、ソース・ABI・生成 ELF の
入口検査、compile と worker の要求同一性を扱う。mock の成功は実機実行や数値
性能の証拠ではない。実際に完了した検証段階は E219 の実験記録に置く。
