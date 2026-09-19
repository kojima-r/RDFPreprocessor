from rdflib import XSD
from rdflib.term import _toPythonMapping
from rdflib.plugins.parsers.ntriples import W3CNTriplesParser

import glob
import os
import rdflib
from multiprocessing import Pool


# Literal の正規化・Python値変換をなるべく抑制
rdflib.NORMALIZE_LITERALS = False
_toPythonMapping.pop(XSD.dateTime, None)
_toPythonMapping.pop(XSD.date, None)


def clean_line(line):
    """
    N-Triples 内の不正な NBSP を除去する。
    例:
      <http://identifiers.org/kegg.compound/C03561\u00a0>
    を
      <http://identifiers.org/kegg.compound/C03561>
    にする。
    """
    return line.replace("\u00a0", "") if "\u00a0" in line else line


class CleanedLineReader:
    """
    一時ファイルを作らず、readline() 時に必要最小限だけ掃除する wrapper。
    """
    def __init__(self, fp):
        self.fp = fp
        self.encoding = getattr(fp, "encoding", "utf-8")

    def readline(self, size=-1):
        line = self.fp.readline(size)
        return clean_line(line) if line else line

    def read(self, size=-1):
        data = self.fp.read(size)
        return clean_line(data) if data else data

    def close(self):
        return self.fp.close()


def safe_text(x):
    return str(x).replace("\r\n", "  ").replace("\n", "  ").replace("\t", "  ")


class TSVSink:
    """
    rdflib parser から流れてきた triple を Graph に貯めず、直接 TSV に書く。
    """
    def __init__(self, out_fp):
        self.write = out_fp.write

    def triple(self, s, p, o):
        self.write(
            type(s).__name__ + "\t" +
            type(p).__name__ + "\t" +
            type(o).__name__ + "\t" +
            safe_text(s) + "\t" +
            safe_text(p) + "\t" +
            safe_text(o) + "\n"
        )


def conv(filename, out_filename):
    with open(filename, "r", encoding="utf-8", errors="replace", buffering=1024 * 1024) as fp, \
         open(out_filename, "w", encoding="utf-8", buffering=1024 * 1024) as ofp:

        parser = W3CNTriplesParser(sink=TSVSink(ofp))
        parser.parse(CleanedLineReader(fp))


def run(args):
    filename, out_filename = args

    try:
        conv(filename, out_filename)
        return filename, out_filename, None
    except Exception as e:
        return filename, out_filename, repr(e)


if __name__ == "__main__":
    data = []

    target = "data01/**/*.nt"

    for filename in glob.glob(target, recursive=True):
        path = os.path.dirname(filename)
        name = os.path.basename(filename)

        path1 = "data02" + path[6:]
        os.makedirs(path1, exist_ok=True)

        filename = path + "/" + name

        name_, _ = os.path.splitext(name)
        out_filename = path1 + "/" + name_ + ".tsv"

        if not os.path.isfile(out_filename):
            print(filename, out_filename)
            data.append((filename, out_filename))
        else:
            print("[EXIST]", out_filename)

    with Pool(8) as p:
        for filename, out_filename, err in p.imap_unordered(run, data, chunksize=1):
            if err:
                print("[ERROR]", filename, "->", out_filename, err)
            else:
                print("[OK]", filename, "->", out_filename)
