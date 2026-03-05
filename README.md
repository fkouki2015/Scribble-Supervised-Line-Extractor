# Scribble-Supervised Line Extractor

機械学習を用いた線画抽出ツールです。
画像に線画領域、非線画領域を手書きで指定することで、AIが学習して線画を抽出します。

## セットアップ方法

### インストール

```bash
$ npm install
$ pip install -r requirements.txt
```

### 開発版の起動

```bash
$ npm run dev
```

### ビルド

```bash
# WIndows:
$ npm run build:win

# MacOS:
$ npm run build:mac
```

## 使い方

### U-Netを用いた方式（低速・高精度）

1. 画像を読み込む
2. 線画領域、非線画領域の一部を手書きで指定する
3. 指定した領域の線画が即座に生成される
4. 「全体の線画を生成」ボタンを押し、AIが線画を抽出するのを待つ
5. 結果を保存する

### Frangiフィルタを用いた方式（高速・低精度）

1. 画像を読み込む
2. 「線画抽出」ボタンを押す
3. しきい値をスライダーで調整
4. 結果を保存する