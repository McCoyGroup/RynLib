import re, os

__all__ = [
    "StringMatcher",
    "MatchList",
    "FileMatcher"
]

class StringMatcher:
    """
    Defines a simple filter that applies to a file and determines whether or not it matches the pattern
    """

    def __init__(self, match_patterns, negative_match = False):
        if isinstance(match_patterns, str):
            pattern = re.compile(match_patterns)
            self.matcher = lambda f, p = pattern: re.match(p, f)
        elif isinstance(match_patterns, re.Pattern):
            self.matcher = lambda f, p = match_patterns: re.match(p, f)
        elif isinstance(match_patterns, StringMatcher):
            self.matcher = match_patterns.matches
        elif callable(match_patterns):
            self.matcher = match_patterns
        else:
            ff = type(self)
            match_patterns = tuple(ff(m) if not isinstance(m, StringMatcher) else m for m in match_patterns)
            self.matcher = lambda f, p = match_patterns: all(m.matches(f) for m in p)

        self.negate = negative_match

    def matches(self, f):
        m = self.matcher(f)
        if self.negate:
            m = not m
        return m

class MatchList(StringMatcher):
    """
    Defines a set of matches that must be matched directly (uses `set` to make this basically a constant time check)
    """

    def __init__(self, *matches, negative_match = False):
        super().__init__(lambda f, m=set(matches): f in m, negative_match = negative_match)

class FileMatcher(StringMatcher):
    """
    Defines a filter that uses StringMatcher to specifically match files
    """

    def __init__(self, match_patterns, negative_match = False, use_basename = False):
        super().__init__(match_patterns, negative_match = negative_match)
        self.use_basename = use_basename

    def matches(self, f):
        return super().matches(f if not self.use_basename else os.path.basename(f)[0])