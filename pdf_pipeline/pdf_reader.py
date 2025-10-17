from __future__ import annotations

import re
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz


class PDFReader:
    """
    read pdfs and extract structured sections tuned for scientific papers
    
    learns heading patterns from abstract/introduction then applies them
    throughout the document for robust section detection
    """

    def __init__(
        self,
        min_characters: int = 300,
        detect_mode: str = "hybrid",
        save_text: bool = False,
        include_references: bool = False,
    ):
        """
        initialize pdf reader with detection parameters
        
        :param min_characters: minimum characters for a valid section body
        :param detect_mode: one of ['layout', 'numeric', 'hybrid']
        :param save_text: if true, saves cleaned section text to data/cache/
        :param include_references: if false, stops extraction at references section
        """
        self.min_characters = min_characters
        self.detect_mode = detect_mode
        self.save_text = save_text
        self.include_references = include_references

    def read_text(self, path: str | Path) -> str:
        """
        extract raw text from a pdf
        
        :param path: filesystem path to pdf
        :return: full plain text of all pages concatenated
        """
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"pdf not found: {pdf_path}")

        doc = fitz.open(pdf_path.as_posix())
        try:
            parts: List[str] = []
            for page in doc:
                text = page.get_text("text")
                if text:
                    parts.append(text)
            return "\n".join(parts)
        finally:
            doc.close()

    def _find_abstract_and_introduction(self, doc: fitz.Document) -> Optional[Dict[str, any]]:
        """
        locate introduction to learn the heading pattern
        introduction is typically the first numbered section (1, I, etc.)
        
        :param doc: opened pymupdf document object
        :return: dictionary with heading pattern info or none if not found
        """
        intro_info = None
        
        # search first 5 pages for the first numbered section
        for page_num in range(min(5, len(doc))):
            page = doc[page_num]
            blocks = page.get_text("dict").get("blocks", [])
            
            for block_idx, block in enumerate(blocks):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    
                    line_text = " ".join(span.get("text", "").strip() for span in spans if span.get("text", "").strip())
                    if not line_text or len(line_text) < 3:
                        continue
                    
                    # skip very long lines (not headings)
                    word_count = len(line_text.split())
                    if word_count > 10:
                        continue
                    
                    # normalize for matching
                    normalized = line_text.replace(" ", "").lower().strip()
                    
                    first_span = spans[0]
                    size = first_span.get("size", 0)
                    flags = first_span.get("flags", 0)
                    is_bold = bool(flags & 2**4)
                    
                    # look for first numbered section that contains "introduction"
                    # patterns: "1 Introduction", "1. Introduction", "I Introduction", "1 INTRODUCTION"
                    intro_patterns = [
                        r'^1\.?\s*introduction',           # 1 Introduction, 1. Introduction
                        r'^i\.?\s*introduction',           # I Introduction, I. Introduction  
                        r'^introduction',                  # Introduction (no number)
                    ]
                    
                    for pattern in intro_patterns:
                        if re.match(pattern, normalized):
                            intro_info = {
                                'size': size,
                                'bold': is_bold,
                                'flags': flags,
                                'has_number': bool(re.match(r'^[\d]+', normalized)) or bool(re.match(r'^[ivx]+', normalized)),
                                'has_roman': bool(re.match(r'^[ivx]+', normalized)),
                            }
                            break
                    
                    if intro_info:
                        break
                
                if intro_info:
                    break
            
            if intro_info:
                break
        
        if not intro_info:
            # fallback: look for any first numbered section (1, 1., I, I.)
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                blocks = page.get_text("dict").get("blocks", [])
                
                for block in blocks:
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        
                        line_text = " ".join(span.get("text", "").strip() for span in spans if span.get("text", "").strip())
                        if not line_text or len(line_text.split()) > 10:
                            continue
                        
                        normalized = line_text.replace(" ", "").lower().strip()
                        first_span = spans[0]
                        
                        # match "1 Something", "1. Something", "I Something"
                        if re.match(r'^(1|i)\.?\s+[a-z]+', normalized):
                            intro_info = {
                                'size': first_span.get("size", 0),
                                'bold': bool(first_span.get("flags", 0) & 2**4),
                                'flags': first_span.get("flags", 0),
                                'has_number': bool(re.match(r'^1', normalized)),
                                'has_roman': bool(re.match(r'^i', normalized)),
                            }
                            print(f"Inferred pattern from: '{line_text}' on page {page_num}")
                            break
                    
                    if intro_info:
                        break
                
                if intro_info:
                    break
        
        if not intro_info:
            print(f"Could not detect heading pattern")
            return None
        
        return {
            'size': intro_info['size'],
            'size_tolerance': 0.5,
            'bold': intro_info['bold'],
            'flags': intro_info['flags'],
            'has_number': intro_info['has_number'],
            'has_roman': intro_info['has_roman'],
        }

    def _matches_heading_pattern(
        self, 
        text: str, 
        size: float, 
        flags: int,
        pattern: Dict[str, any],
        prev_y: Optional[float],
        curr_y: float,
    ) -> Tuple[bool, float]:
        """
        check if text matches the learned heading pattern
        
        :param text: text to check
        :param size: font size
        :param flags: font flags
        :param pattern: learned pattern from abstract/introduction
        :param prev_y: previous y position
        :param curr_y: current y position
        :return: (matches, confidence)
        """
        if not text or len(text) < 3:
            return False, 0.0
        
        text_stripped = text.strip()
        
        # reject if text has single-character words (spacing issue in PDF)
        # e.g. "I NTRODUCTION" or "P RELIMINARY" should be rejected
        words = text_stripped.split()
        single_char_words = sum(1 for w in words if len(w) == 1)
        
        # if more than 30% of words are single characters, it's a spacing issue
        if len(words) > 1 and single_char_words / len(words) > 0.3:
            return False, 0.0
        
        # basic filters
        if text_stripped.startswith(('•', '◦', '▪', '▫', '‣', '-', '*', '·')):
            return False, 0.0
        
        if re.match(r"^(figure|fig\.|table|tab\.)\s*\d+", text_stripped, re.I):
            return False, 0.0
        
        if len(text_stripped) > 120:
            return False, 0.0
        
        if text_stripped[0].islower() and not re.match(r"^\d", text_stripped):
            return False, 0.0
        
        if text_stripped.endswith((',', ';', 'and', 'or', 'the', 'a', 'an', 'of', 'in', 'to', 'for', 'with', 'by', 'at')):
            return False, 0.0
        
        if re.match(r'^(as shown|as of|in the|on the|for the|with the|by the|from the|to the|all datasets|despite)', 
                text_stripped.lower()):
            return False, 0.0
        
        if len(words) > 15 or len(words) < 1:
            return False, 0.0
        
        confidence = 0.0
        
        # check font size match
        size_diff = abs(size - pattern['size'])
        if size_diff <= pattern['size_tolerance']:
            confidence += 0.5
        elif size_diff <= pattern['size_tolerance'] * 2:
            confidence += 0.3
        else:
            return False, 0.0
        
        # check bold match
        is_bold = bool(flags & 2**4)
        if is_bold == pattern['bold']:
            confidence += 0.3
        
        # check numbering pattern (DON'T penalize if missing, just reward if present)
        has_number = bool(re.match(r'^\d+\.?\s+', text_stripped))
        has_roman = bool(re.match(r'^[ivxlcdm]+\.?\s+', text_stripped, re.I))
        
        if pattern['has_number'] and has_number:
            confidence += 0.3
        
        if pattern['has_roman'] and has_roman:
            confidence += 0.3
        
        # check capitalization
        if text_stripped.isupper() and len(words) <= 6:
            confidence += 0.2
        elif words and words[0][0].isupper():
            confidence += 0.1
        
        # check spacing (headings usually have vertical space before them)
        if prev_y is not None:
            gap = curr_y - prev_y
            if gap >= 20:
                confidence += 0.2
            elif gap >= 10:
                confidence += 0.1
        
        # penalize body text indicators
        body_indicators = ['the', 'and', 'or', 'but', 'however', 'therefore', 'thus', 'moreover', 'which', 'that', 'this', 'these']
        body_count = sum(1 for w in words if w.lower() in body_indicators)
        if body_count >= 2:
            confidence -= 0.4
        
        # penalize very short single-word headings (likely not real sections)
        if len(words) == 1 and len(text_stripped) < 5 and not has_number:
            confidence -= 0.3
        
        return confidence >= 0.6, confidence

    def _should_stop_at_section(self, section_name: str) -> bool:
        """
        check if we should stop extraction at this section
        
        :param section_name: name of the section
        :return: true if we should stop extraction
        """
        if not self.include_references:
            if re.match(r'^(\d+\.?\s+)?(references|bibliography|appendix)', section_name.strip().lower()):
                return True
        return False

    def _extract_by_layout(self, path: Path) -> Dict[str, str]:
        """
        detect sections by learning pattern from abstract/introduction
        
        :param path: pdf path
        :return: mapping of section title to text
        """
        doc = fitz.open(path.as_posix())
        sections: Dict[str, List[str]] = {}
        current_section = "Document"
        sections[current_section] = []
        should_stop = False

        try:
            pattern = self._find_abstract_and_introduction(doc)
            
            if not pattern:
                doc.close()
                text = self.read_text(path)
                return self._extract_by_numbering(text, path)
            
            # sample first 100 lines and check for single-character word ratio
            spacing_issue_detected = False
            sample_lines = 0
            total_single_char = 0
            total_words = 0
            
            for page_num in range(min(2, len(doc))):
                page = doc[page_num]
                blocks = page.get_text("dict").get("blocks", [])
                # check first 20 blocks
                for block in blocks[:20]:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                words = text.split()
                                total_words += len(words)
                                total_single_char += sum(1 for w in words if len(w) == 1)
                                sample_lines += 1
                                
                                if sample_lines >= 100:
                                    break
                        if sample_lines >= 100:
                            break
                    if sample_lines >= 100:
                        break
                if sample_lines >= 100:
                    break
            
            # if more than 15% of words are single characters, document has spacing issues
            if total_words > 0:
                single_char_ratio = total_single_char / total_words
                
                if single_char_ratio > 0.15:
                    spacing_issue_detected = True
                    doc.close()
                    text = self.read_text(path)
                    return self._extract_by_numbering(text, path)
            
            prev_y = None
            heading_buffer: List[str] = []
            heading_confidence: List[float] = []
            last_was_heading = False

            for page in doc:
                if should_stop:
                    break
                
                blocks = page.get_text("dict").get("blocks", [])
                for block in blocks:
                    if should_stop:
                        break
                    
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        
                        first_span = spans[0]
                        text = first_span.get("text", "").strip()
                        
                        if not text:
                            continue
                        
                        y0 = first_span.get("bbox", [0, 0, 0, 0])[1]
                        size = first_span.get("size", 0)
                        flags = first_span.get("flags", 0)
                        
                        # collect full line
                        full_line_parts = [text]
                        for span in spans[1:]:
                            span_text = span.get("text", "").strip()
                            if span_text:
                                full_line_parts.append(span_text)
                        full_line = " ".join(full_line_parts)
                        
                        is_heading, confidence = self._matches_heading_pattern(
                            full_line, size, flags, pattern, prev_y, y0
                        )
                        
                        if is_heading:
                            if heading_buffer and not last_was_heading:
                                merged = " ".join(heading_buffer).strip()
                                merged = re.sub(r'\s+', ' ', merged)
                                
                                if max(heading_confidence) >= 0.7:
                                    current_section = merged
                                    if current_section not in sections:
                                        sections[current_section] = []
                                    
                                    if self._should_stop_at_section(current_section):
                                        should_stop = True
                                
                                heading_buffer.clear()
                                heading_confidence.clear()
                            
                            heading_buffer.append(full_line)
                            heading_confidence.append(confidence)
                            last_was_heading = True
                            prev_y = y0
                            continue
                        
                        # body text
                        if heading_buffer:
                            merged = " ".join(heading_buffer).strip()
                            merged = re.sub(r'\s+', ' ', merged)
                            
                            if heading_confidence and max(heading_confidence) >= 0.7:
                                current_section = merged
                                if current_section not in sections:
                                    sections[current_section] = []
                                
                                if self._should_stop_at_section(current_section):
                                    should_stop = True
                            else:
                                for h in heading_buffer:
                                    sections[current_section].append(h)
                            
                            heading_buffer.clear()
                            heading_confidence.clear()
                        
                        if should_stop:
                            break
                        
                        for span in spans:
                            span_text = span.get("text", "").strip()
                            if span_text:
                                sections[current_section].append(span_text)
                        
                        last_was_heading = False
                        prev_y = y0

            return self._clean_sections(sections, path)
        finally:
            doc.close()

    def _extract_by_numbering(self, text: str, path: Path) -> Dict[str, str]:
        """
        detect sections by numbered or distinctive text patterns (fallback)
        
        :param text: raw pdf text
        :param path: pdf path
        :return: mapping of section title to text
        """
        heading_pattern = re.compile(
            r"(?m)^("
            r"\d+\.?\s+[A-Z][A-Za-z\s]{2,60}$|"
            r"[A-Z][A-Z\s]{10,50}$"
            r")"
        )
        
        lines = text.splitlines()
        sections: Dict[str, List[str]] = {}
        current_section = "Document"
        sections[current_section] = []
        should_stop = False

        for line in lines:
            if should_stop:
                break
            
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            match = heading_pattern.match(line_stripped)
            
            if match:
                words = line_stripped.split()
                if 1 <= len(words) <= 12:
                    heading = re.sub(r'\s+', ' ', line_stripped)
                    current_section = heading
                    if current_section not in sections:
                        sections[current_section] = []
                    
                    if self._should_stop_at_section(current_section):
                        should_stop = True
                    continue
            
            sections[current_section].append(line)

        return self._clean_sections(sections, path)

    def _clean_sections(self, raw_sections: Dict[str, List[str]], path: Path) -> Dict[str, str]:
        """
        clean, merge, and optionally save extracted sections
        
        :param raw_sections: raw mapping of section name to text list
        :param path: pdf path
        :return: cleaned mapping of section name to text
        """
        sections: Dict[str, str] = {
            name.strip(): "\n".join(body).strip() 
            for name, body in raw_sections.items()
            if name.strip()
        }

        filtered = {
            k: v
            for k, v in sections.items()
            if len(v) >= self.min_characters
        }
        
        filtered = {
            k: v for k, v in filtered.items()
            if not re.match(r'^(page \d+|\d+|https?://|www\.|\[\d+\])$', k, re.I)
            and len(k) >= 3
            and not k.startswith(('•', '◦', '▪', '▫', '‣', '-', '*', '·'))
            and not k.lower().startswith(('as shown', 'as of', 'in the', 'on the', 'for the'))
            and not re.search(r'\[\s*\d+\s*\]', k)
        }

        # merge near-duplicates
        names = list(filtered.keys())
        merged: Dict[str, str] = {}
        skip = set()
        
        for i, n1 in enumerate(names):
            if n1 in skip:
                continue
            merged_text = filtered[n1]
            for n2 in names[i + 1:]:
                if n2 in skip:
                    continue
                ratio = difflib.SequenceMatcher(None, n1.lower(), n2.lower()).ratio()
                if ratio > 0.85:
                    merged_text += "\n" + filtered[n2]
                    skip.add(n2)
            merged[n1] = merged_text

        if self.save_text:
            cache_dir = Path("data/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_path = cache_dir / f"{path.stem}_sections.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                for name, text in merged.items():
                    f.write(f"## {name}\n{text}\n\n")

        return merged

    def extract_sections(self, path: str | Path) -> Dict[str, str]:
        """
        extract structured sections from a pdf
        
        :param path: filesystem path to pdf
        :return: mapping of section title to text
        """
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"pdf not found: {pdf_path}")

        if self.detect_mode == "layout":
            return self._extract_by_layout(pdf_path)

        if self.detect_mode == "numeric":
            text = self.read_text(pdf_path)
            return self._extract_by_numbering(text, pdf_path)

        try:
            sections = self._extract_by_layout(pdf_path)
            if len(sections) <= 1:
                text = "\n".join(sections.values())
                return self._extract_by_numbering(text, pdf_path)
            return sections
        except Exception:
            text = self.read_text(pdf_path)
            return self._extract_by_numbering(text, pdf_path)
