#!/bin/bash

# 異常終了するまで無限ループ
while true; do
    echo "$(date) プログラムを開始します。"
    
    # --- ここに実行したいコマンドを記述 ---
    python 01download.py
    # ----------------------------------

    # コマンドの戻り値（$?）をチェック
    if [ $? -ne 0 ]; then
        echo "$(date) エラーが発生しました。1時間後に再起動します。"
        sleep 7200
    else
        echo "$(date) プログラムが正常終了しました。"
        break # 正常終了したらループを抜ける
    fi
done
