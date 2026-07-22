# =============================================================================
# Shared plotting-test assertions (plotrule Testing Rules 1-7).
#
# Dependency-FREE: the package pulls in no JSON library (plotrule A12 forbids
# adding one), so the strict-JSON validator below is hand-rolled. This file is a
# plain `include`d script (NOT a module), so every plotting test lane — the native
# test/plotting/* files AND the scattered domain test files — can pull it in with
#     include(joinpath(@__DIR__, "..", "plotting", "plot_test_helpers.jl"))
# and share `check_plot`, `assert_strict_json`, `assert_escapes`, etc.
#
# The canonical hostile name exercises all three output sinks (A7/A8): a JS-string
# terminator ("), an HTML tag opener (<c>), an entity (&), a script terminator
# (</script>) and a trailing space.
# =============================================================================

using Test

const HOSTILE_NAME = "a\"b<c>&d</script> e"

# -----------------------------------------------------------------------------
# Rule 1 (smoke) + A12 (self-containment)
# -----------------------------------------------------------------------------

"""
    assert_self_contained(html; allow=String[])

Fail on any `src=`/`href=` pointing at an off-host `http(s)://` URL that is not in
`allow`. Supersedes the old `occursin("d3.min.js")` check, which passed on the CDN
URL and so could never catch a regression (plotrule A12).
"""
function assert_self_contained(html::AbstractString; allow::Vector{String}=String[])
    for m in eachmatch(r"(?:src|href)\s*=\s*[\"']([^\"']+)[\"']", html)
        u = m.captures[1]
        (startswith(u, "http://") || startswith(u, "https://")) || continue
        @test u in allow
    end
    return nothing
end

"""
    check_plot(p; min_size=500, allow_urls=String[])

Smoke + self-containment gate for a `PlotOutput`: it is a `PlotOutput`, is large
enough, is a real document, builds an SVG at runtime, embeds no external resource,
and — when saved — begins with `<!DOCTYPE html>` (Testing Rules 1, A12).
"""
function check_plot(p::PlotOutput; min_size::Int=500, allow_urls::Vector{String}=String[])
    @test p isa PlotOutput
    @test length(p.html) >= min_size
    @test occursin("<!DOCTYPE html>", p.html)
    @test occursin("append('svg')", p.html)             # D3 builds the svg at runtime
    assert_self_contained(p.html; allow=allow_urls)
    tmp = tempname() * ".html"
    save_plot(p, tmp)
    @test startswith(strip(read(tmp, String)), "<!DOCTYPE html>")
    rm(tmp; force=true)
    return p
end

# -----------------------------------------------------------------------------
# Rule 6 (strict JSON): hand-rolled extractor + recursive-descent validator
# -----------------------------------------------------------------------------

# Balance []/{} from the opening bracket at `start`, respecting string literals and
# escapes. Returns the index of the matching close bracket (0 if unbalanced).
function _tj_balance(s::AbstractString, start::Int)
    depth = 0
    in_str = false
    esc = false
    i = start
    n = lastindex(s)
    while i <= n
        c = s[i]
        if in_str
            if esc
                esc = false
            elseif c == '\\'
                esc = true
            elseif c == '"'
                in_str = false
            end
        elseif c == '"'
            in_str = true
        elseif c == '[' || c == '{'
            depth += 1
        elseif c == ']' || c == '}'
            depth -= 1
            depth == 0 && return i
        end
        i = nextind(s, i)
    end
    return 0
end

"""
    extract_json_blocks(html) -> Vector{Pair{String,String}}

Return `(name, literal)` for every `const NAME = [..]|{..};` the renderers emit —
`data`, `series`, `bands`, `refLines`, `rowLabels`, `colLabels` (line/area/bar/heat).
Scalar consts (`mode='stacked'`, `W`, `margin`, …) are ignored.
"""
function extract_json_blocks(html::AbstractString)::Vector{Pair{String,String}}
    names = ["data", "series", "bands", "refLines", "rowLabels", "colLabels"]
    out = Pair{String,String}[]
    for nm in names
        anchor = "const " * nm * " ="
        pos = firstindex(html)
        while true
            r = findnext(anchor, html, pos)
            r === nothing && break
            j = nextind(html, last(r))                  # char after '='
            while j <= lastindex(html) && isspace(html[j])
                j = nextind(html, j)
            end
            pos = j
            if j <= lastindex(html) && (html[j] == '[' || html[j] == '{')
                k = _tj_balance(html, j)
                if k != 0
                    push!(out, nm => String(html[j:k]))
                    pos = nextind(html, k)
                end
            end
            pos > lastindex(html) && break
        end
    end
    return out
end

_tj_skip_ws(s, i) = (while i <= lastindex(s) && isspace(s[i]); i = nextind(s, i); end; i)

function _tj_expect(s, i, tok)
    j = i
    for tc in tok
        (j <= lastindex(s) && s[j] == tc) || error("expected '$tok' at index $i")
        j = nextind(s, j)
    end
    return j
end

function _tj_parse_string(s, i)
    io = IOBuffer()
    j = nextind(s, i)                                   # past opening quote
    while j <= lastindex(s)
        c = s[j]
        if c == '"'
            return (String(take!(io)), nextind(s, j))
        elseif c == '\\'
            j = nextind(s, j)
            j <= lastindex(s) || error("dangling escape")
            e = s[j]
            if     e == '"';  write(io, '"')
            elseif e == '\\'; write(io, '\\')
            elseif e == '/';  write(io, '/')
            elseif e == 'n';  write(io, '\n')
            elseif e == 'r';  write(io, '\r')
            elseif e == 't';  write(io, '\t')
            elseif e == 'b';  write(io, '\b')
            elseif e == 'f';  write(io, '\f')
            elseif e == 'u'
                hexs = ""
                for _ in 1:4
                    j = nextind(s, j)
                    j <= lastindex(s) || error("bad \\u escape")
                    hexs *= s[j]
                end
                write(io, Char(parse(UInt16, hexs; base=16)))
            else
                error("invalid escape \\$e")
            end
            j = nextind(s, j)
        elseif c < ' '
            error("unescaped control char in string")
        else
            write(io, c)
            j = nextind(s, j)
        end
    end
    error("unterminated string")
end

function _tj_parse_number(s, i)
    start = i
    j = i
    if j <= lastindex(s) && s[j] == '-'
        j = nextind(s, j)
    end
    (j <= lastindex(s) && (s[j] == 'I' || s[j] == 'N')) &&
        error("bare Infinity/NaN is not valid JSON")
    while j <= lastindex(s)
        c = s[j]
        (('0' <= c <= '9') || c in ('.', 'e', 'E', '+', '-')) || break
        j = nextind(s, j)
    end
    return (parse(Float64, s[start:prevind(s, j)]), j)
end

function _tj_parse_array(s, i)
    arr = Any[]
    j = _tj_skip_ws(s, nextind(s, i))
    if j <= lastindex(s) && s[j] == ']'
        return (arr, nextind(s, j))
    end
    while true
        v, j = _tj_parse_value(s, j)
        push!(arr, v)
        j = _tj_skip_ws(s, j)
        j <= lastindex(s) || error("unterminated array")
        if s[j] == ','
            j = _tj_skip_ws(s, nextind(s, j))
        elseif s[j] == ']'
            return (arr, nextind(s, j))
        else
            error("expected ',' or ']' in array")
        end
    end
end

function _tj_parse_object(s, i)
    obj = Dict{String,Any}()
    seen = Set{String}()
    j = _tj_skip_ws(s, nextind(s, i))
    if j <= lastindex(s) && s[j] == '}'
        return (obj, nextind(s, j))
    end
    while true
        j = _tj_skip_ws(s, j)
        (j <= lastindex(s) && s[j] == '"') || error("expected string key")
        key, j = _tj_parse_string(s, j)
        key in seen && error("duplicate object key \"$key\"")   # the bridge bug (Rule 6)
        push!(seen, key)
        j = _tj_skip_ws(s, j)
        (j <= lastindex(s) && s[j] == ':') || error("expected ':'")
        v, j = _tj_parse_value(s, nextind(s, j))
        obj[key] = v
        j = _tj_skip_ws(s, j)
        j <= lastindex(s) || error("unterminated object")
        if s[j] == ','
            j = nextind(s, j)
        elseif s[j] == '}'
            return (obj, nextind(s, j))
        else
            error("expected ',' or '}' in object")
        end
    end
end

"""
    _tj_parse_value(s, i) -> (value, next_index)

Recursive-descent parser over the six JSON value forms. THROWS on duplicate object
keys (Rule 6) and on bare `NaN`/`Infinity`/`-Infinity`/`undefined` outside a string
(Rule 4 — those must serialize to `null`).
"""
function _tj_parse_value(s, i)
    i = _tj_skip_ws(s, i)
    i <= lastindex(s) || error("unexpected end of JSON")
    c = s[i]
    if     c == '{'  return _tj_parse_object(s, i)
    elseif c == '['  return _tj_parse_array(s, i)
    elseif c == '"'  return _tj_parse_string(s, i)
    elseif c == 't'  return (true,    _tj_expect(s, i, "true"))
    elseif c == 'f'  return (false,   _tj_expect(s, i, "false"))
    elseif c == 'n'  return (nothing, _tj_expect(s, i, "null"))
    elseif c == '-' || ('0' <= c <= '9') return _tj_parse_number(s, i)
    elseif c == 'N' || c == 'I' error("bare NaN/Infinity is not valid JSON")
    elseif c == 'u'                     error("bare undefined is not valid JSON")
    else error("unexpected char '$c' in JSON")
    end
end

"""
    assert_strict_json(literal) -> parsed value

Parse a JSON literal, failing the enclosing `@test` set on any malformed input,
duplicate object key, bare `NaN`/`Infinity`/`undefined`, or trailing junk.
"""
function assert_strict_json(literal::AbstractString)
    err = nothing
    val = nothing
    try
        val, pos = _tj_parse_value(literal, firstindex(literal))
        pos = _tj_skip_ws(literal, pos)
        pos <= lastindex(literal) && (err = "trailing content after JSON at $pos")
    catch e
        err = sprint(showerror, e)
    end
    if err !== nothing
        @info "assert_strict_json failure" err prefix=first(literal, 160)
    end
    @test err === nothing
    return val
end

"Extract every embedded JSON literal from a PlotOutput and strict-parse each."
function assert_all_json_valid(p::PlotOutput)
    blocks = extract_json_blocks(p.html)
    for (_, lit) in blocks
        assert_strict_json(lit)
    end
    return blocks
end

# -----------------------------------------------------------------------------
# Rule 2 (structural content)
# -----------------------------------------------------------------------------

# Names/count come from the FIRST `series` block (single-panel plots have one).
function _tj_first_series(html::AbstractString)
    for (nm, lit) in extract_json_blocks(html)
        nm == "series" || continue
        val, _ = _tj_parse_value(lit, firstindex(lit))
        return val isa AbstractVector ? val : Any[]
    end
    return Any[]
end

series_names(html::AbstractString)::Vector{String} =
    String[string(get(o, "name", "")) for o in _tj_first_series(html) if o isa AbstractDict]

series_count(html::AbstractString)::Int = length(_tj_first_series(html))

panel_titles(html::AbstractString)::Vector{String} =
    String[String(m.captures[1]) for m in eachmatch(r"<div class=\"panel-title\">(.*?)</div>"s, html)]

# -----------------------------------------------------------------------------
# Rule 5 (escaping round-trip, A7/A8)
# -----------------------------------------------------------------------------

"""
    assert_escapes(p)

The hostile name must survive at every sink: no raw `<c>` inside any `panel-title`
element (HTML sink), no raw `<c>` inside any embedded data literal (`<` → `\\u003c`,
JS/JSON sink), and every embedded literal still parses as strict JSON.
"""
function assert_escapes(p::PlotOutput)
    for t in panel_titles(p.html)
        @test !occursin("<c>", t)                        # HTML-text sink escaped
    end
    for (_, lit) in extract_json_blocks(p.html)
        @test !occursin("<c>", lit)                      # JS/JSON string sink escaped
    end
    assert_all_json_valid(p)
    return nothing
end

# -----------------------------------------------------------------------------
# Rule 4 (NaN → null)
# -----------------------------------------------------------------------------

"""
    assert_nan_becomes_null(p)

Assert that non-finite input serialized to `null` and NO bare `NaN`/`Infinity`
token survives into any embedded data literal. Scans only the extracted JSON blocks
— NOT `p.html`, which inlines the vendored D3 library (whose own minified source
legitimately contains `NaN`/`:NaN` tokens, plotrule A12/PLT-01). Call on a plot whose
input contained NaN/Inf.
"""
function assert_nan_becomes_null(p::PlotOutput)
    blocks = extract_json_blocks(p.html)
    @test !isempty(blocks)
    sawnull = false
    for (_, lit) in blocks
        @test !occursin(r"[:\[,]\s*NaN", lit)   # NaN never leaks into a data literal
        assert_strict_json(lit)                  # strict-JSON also rejects bare NaN/Inf
        sawnull |= occursin("null", lit)
    end
    @test sawnull                                 # the NaN/Inf input became a JSON null
    return nothing
end
