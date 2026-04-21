from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

@dataclass
class FastaRecord:
    header: str
    sequence: str

    @property
    def accession(self) -> str:
        return self.header.split()[0] if self.header else ''

def read_fasta(path: str | Path) -> List[FastaRecord]:
    text = Path(path).read_text()
    records: List[FastaRecord] = []
    header = None
    parts: List[str] = []
    for line in text.splitlines():
        if line.startswith('>'):
            if header is not None:
                records.append(FastaRecord(header, ''.join(parts).upper()))
            header = line[1:].strip()
            parts = []
        else:
            parts.append(line.strip())
    if header is not None:
        records.append(FastaRecord(header, ''.join(parts).upper()))
    return records

@dataclass
class CDSFeature:
    start: int
    end: int
    strand: str
    intervals: List[Tuple[int, int]] = field(default_factory=list)
    gene: str = ''
    product: str = ''

    @property
    def length(self) -> int:
        return sum((b - a + 1 for a, b in self.intervals))

@dataclass
class GenBankRecord:
    locus: str
    length: int
    features: List[CDSFeature]
_LOC_RE = re.compile('(\\d+)\\.\\.(\\d+)')

def parse_location(loc: str) -> Tuple[str, List[Tuple[int, int]]]:
    strand = '+'
    s = loc
    if s.startswith('complement('):
        strand = '-'
        s = s[len('complement('):-1]
    if s.startswith('join('):
        s = s[len('join('):-1]
    intervals: List[Tuple[int, int]] = []
    for m in _LOC_RE.finditer(s):
        intervals.append((int(m.group(1)), int(m.group(2))))
    return (strand, intervals)

def read_genbank(path: str | Path) -> GenBankRecord:
    text = Path(path).read_text()
    locus_match = re.search('^LOCUS\\s+(\\S+)\\s+(\\d+)\\s+bp', text, flags=re.M)
    if not locus_match:
        raise ValueError('not a GenBank file (no LOCUS line)')
    locus = locus_match.group(1)
    length = int(locus_match.group(2))
    if 'FEATURES' not in text:
        return GenBankRecord(locus=locus, length=length, features=[])
    feat_text = text.split('FEATURES', 1)[1].split('ORIGIN', 1)[0]
    features: List[CDSFeature] = []
    entries = re.split('\\n(?=     \\S)', feat_text)
    for entry in entries:
        stripped = entry.strip()
        if not stripped.startswith('CDS'):
            continue
        first_line_end = entry.find('\n')
        first_line = entry[:first_line_end] if first_line_end != -1 else entry
        lines = entry.splitlines()
        loc_parts = [lines[0].split('CDS', 1)[1].strip()]
        i = 1
        while i < len(lines) and lines[i].strip().startswith('/') is False:
            loc_parts.append(lines[i].strip())
            i += 1
        loc = ''.join(loc_parts)
        strand, intervals = parse_location(loc)
        if not intervals:
            continue
        start = min((a for a, _ in intervals))
        end = max((b for _, b in intervals))
        gene = ''
        product = ''
        g_match = re.search('/gene="([^"]+)"', entry)
        if g_match:
            gene = g_match.group(1)
        p_match = re.search('/product="([^"]+)"', entry, re.DOTALL)
        if p_match:
            product = re.sub('\\s+', ' ', p_match.group(1)).strip()
        features.append(CDSFeature(start=start, end=end, strand=strand, intervals=intervals, gene=gene, product=product))
    return GenBankRecord(locus=locus, length=length, features=features)