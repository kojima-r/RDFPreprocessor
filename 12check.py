def print_first_and_last(filename, encoding="utf-8"):
    with open(filename, "rb") as f:
        # 最初の行
        first_line = f.readline().decode(encoding).rstrip()

        # 最終行（後ろから読む）
        # ファイル末尾から後ろに向かって改行を2つ探す
        newline_count = 0
        f.seek(-1, 2)  # ファイル末尾

        while f.tell() > 0:
            if f.read(1) == b'\n':
                newline_count += 1
                if newline_count == 2:
                    break
            f.seek(-2, 1)

        last_line = f.readline().decode(encoding).rstrip()

    #print(">>",first_line)
    #print(">>",last_line)
    arr_first=first_line.split("\t")
    arr_last=last_line.split("\t")
    i_first=int(arr_first[0])
    i_last=int(arr_last[0])
    print(i_first, i_last)
    return i_first, i_last
import glob
import os
if __name__ == "__main__":
    target="data06_uniq/*.graph.tsv"
    n1_list=[]
    for src_filename in glob.glob(target,recursive=True):
        if os.path.getsize(src_filename)>10:
            print(src_filename)
            e1,e2=print_first_and_last(src_filename)
            n1_list.append(e1)
            n1_list.append(e2)
    print("===")
    print(min(n1_list),max(n1_list))
