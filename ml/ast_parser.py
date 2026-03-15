from __future__ import annotations

import re
from dataclasses import dataclass

CODE_KEYWORDS = {
    "python": frozenset(
        {
            "def",
            "class",
            "import",
            "from",
            "return",
            "pass",
            "yield",
            "raise",
            "try",
            "except",
            "finally",
            "async",
            "await",
            "with",
        }
    ),
    "javascript": frozenset(
        {
            "function",
            "class",
            "const",
            "let",
            "var",
            "import",
            "export",
            "return",
            "async",
            "await",
            "try",
            "catch",
            "finally",
            "throw",
            "typeof",
            "instanceof",
        }
    ),
    "typescript": frozenset(
        {
            "interface",
            "type",
            "implements",
            "enum",
            "readonly",
            "declare",
            "namespace",
            "function",
            "class",
            "const",
            "let",
            "var",
            "import",
            "export",
            "return",
        }
    ),
    "java": frozenset(
        {
            "class",
            "interface",
            "public",
            "private",
            "protected",
            "static",
            "final",
            "void",
            "return",
            "throws",
            "try",
            "catch",
            "finally",
            "import",
            "package",
        }
    ),
    "c": frozenset(
        {
            "int",
            "char",
            "float",
            "double",
            "void",
            "return",
            "if",
            "else",
            "for",
            "while",
            "do",
            "switch",
            "case",
            "break",
            "continue",
            "struct",
            "typedef",
            "#include",
            "#define",
        }
    ),
    "cpp": frozenset(
        {
            "class",
            "public",
            "private",
            "protected",
            "virtual",
            "override",
            "template",
            "typename",
            "int",
            "char",
            "float",
            "double",
            "void",
            "return",
            "namespace",
            "using",
            "#include",
        }
    ),
    "csharp": frozenset(
        {
            "class",
            "interface",
            "public",
            "private",
            "protected",
            "internal",
            "static",
            "readonly",
            "void",
            "return",
            "async",
            "await",
            "try",
            "catch",
            "finally",
            "using",
            "namespace",
        }
    ),
}

STDLIB_PER_LANGUAGE = {
    "python": frozenset(
        {
            "print",
            "len",
            "range",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "open",
            "super",
            "self",
            "None",
            "True",
            "False",
            "Exception",
        }
    ),
    "javascript": frozenset(
        {
            "console",
            "Math",
            "Array",
            "Object",
            "Promise",
            "fetch",
            "document",
            "window",
            "JSON",
            "React",
            "Component",
            "useState",
            "useEffect",
            "useCallback",
            "useMemo",
            "useRef",
        }
    ),
    "typescript": frozenset(
        {
            "console",
            "Math",
            "Array",
            "Object",
            "Promise",
            "fetch",
            "document",
            "window",
            "JSON",
            "React",
            "Component",
            "useState",
            "useEffect",
            "useCallback",
            "useMemo",
            "useRef",
        }
    ),
    "java": frozenset(
        {
            "String",
            "Integer",
            "System",
            "List",
            "Map",
            "ArrayList",
            "HashMap",
            "Optional",
            "Override",
        }
    ),
    "c": frozenset(
        {
            "printf",
            "scanf",
            "malloc",
            "free",
            "sizeof",
            "std",
            "cout",
            "cin",
            "endl",
            "vector",
            "nullptr",
        }
    ),
    "cpp": frozenset(
        {
            "printf",
            "scanf",
            "malloc",
            "free",
            "sizeof",
            "std",
            "cout",
            "cin",
            "endl",
            "vector",
            "nullptr",
        }
    ),
    "csharp": frozenset(
        {
            "Console",
            "string",
            "int",
            "var",
            "List",
            "Dictionary",
            "IEnumerable",
            "Task",
            "async",
            "await",
        }
    ),
}


@dataclass
class ASTSpan:
    text: str
    start: int
    end: int
    pillar: int
    node_type: str


class ASTParser:
    @staticmethod
    def detect_language(code: str) -> str:
        scores = {lang: 0 for lang in CODE_KEYWORDS}
        tokens = re.split(r"\s+", code)
        for token in tokens:
            stripped = token.strip("():{}[];,.")
            for lang, keywords in CODE_KEYWORDS.items():
                if stripped in keywords:
                    scores[lang] += 1

        if "interface " in code or ": string" in code or ": number" in code:
            scores["typescript"] += 3

        max_score = max(scores.values())
        if max_score == 0:
            return "python"

        for lang, score in scores.items():
            if score == max_score:
                return lang
        return "python"

    @staticmethod
    def is_code_payload(text: str, threshold: float = 0.03) -> bool:
        tokens = text.split()
        if not tokens:
            return False

        hits = 0
        all_keywords = set().union(*CODE_KEYWORDS.values())
        for token in tokens:
            stripped = token.strip("():{}[];")
            if stripped in all_keywords:
                hits += 1

        return hits / len(tokens) >= threshold

    @staticmethod
    def extract_pillars(code: str, language: str) -> list[ASTSpan]:
        try:
            return ASTParser._extract_tree_sitter(code, language)
        except ImportError:
            return ASTParser._extract_heuristic(code, language)
        except Exception:
            return ASTParser._extract_heuristic(code, language)

    @staticmethod
    def _extract_tree_sitter(code: str, language: str) -> list[ASTSpan]:
        # Implementation deferred to full tree-sitter integration
        raise ImportError("tree_sitter not fully configured")

    @staticmethod
    def _extract_heuristic(code: str, language: str) -> list[ASTSpan]:
        spans: list[ASTSpan] = []
        stdlib = STDLIB_PER_LANGUAGE.get(language, frozenset())

        patterns = [
            (r"\bdef\s+(\w+)\s*\(", 1, "function_definition"),
            (r"\bclass\s+(\w+)[\s:(]", 1, "class_definition"),
            (r"\bfunction\s+(\w+)\s*\(", 1, "function_declaration"),
            (r"\binterface\s+(\w+)[\s{<]", 1, "interface_declaration"),
        ]

        for pattern, pillar, node_type in patterns:
            for match in re.finditer(pattern, code):
                ident = match.group(1)
                if ident not in stdlib:
                    spans.append(
                        ASTSpan(
                            text=ident,
                            start=match.start(1),
                            end=match.end(1),
                            pillar=pillar,
                            node_type=node_type,
                        )
                    )

        url_pattern = r'(["\'])([^"\']*(?:internal|corp|\.local|intranet)[^"\']*)\1'
        for match in re.finditer(url_pattern, code):
            url = match.group(2)
            spans.append(
                ASTSpan(
                    text=url,
                    start=match.start(2),
                    end=match.end(2),
                    pillar=3,
                    node_type="string_literal",
                )
            )

        spans.sort(key=lambda s: s.start)
        return spans
