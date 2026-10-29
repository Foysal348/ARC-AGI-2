/* E23 P2: the C23 integer-parsing entry points glibc 2.38 added (__isoc23_strto*), for glibc 2.35.
   The Merlin release .so files call them only through GCC 13 + glibc 2.38 headers; C23 differs from C11 only in
   accepting a "0b"/"0B" prefix with base 0 or 2, which these wrappers add. Built without a version script, so its
   symbols are unversioned (they satisfy the weakened GLIBC_2.38 references, see glibc235_fix.py). LD_PRELOAD it. */
#include <stdlib.h>
#include <ctype.h>

static const char *bin_body(const char *s, int base, int *neg, const char **start)
{
    const char *p = s;
    while (isspace((unsigned char)*p))
        p++;
    *start = p;
    *neg = 0;
    if (*p == '+' || *p == '-') {
        *neg = (*p == '-');
        p++;
    }
    if ((base == 0 || base == 2) && p[0] == '0' && (p[1] == 'b' || p[1] == 'B') && (p[2] == '0' || p[2] == '1'))
        return p + 2;
    return NULL;
}

#define DEFINE(NAME, T, F)                                                                 \
    T NAME(const char *s, char **end, int base)                                            \
    {                                                                                      \
        int neg;                                                                           \
        const char *start, *b = bin_body(s, base, &neg, &start);                           \
        if (b) {                                                                           \
            T v = (T)F(b, end, 2);                                                         \
            return neg ? (T)(0 - v) : v;                                                   \
        }                                                                                  \
        return F(s, end, base);                                                            \
    }

DEFINE(__isoc23_strtol, long, strtol)
DEFINE(__isoc23_strtoll, long long, strtoll)
DEFINE(__isoc23_strtoul, unsigned long, strtoul)
DEFINE(__isoc23_strtoull, unsigned long long, strtoull)
