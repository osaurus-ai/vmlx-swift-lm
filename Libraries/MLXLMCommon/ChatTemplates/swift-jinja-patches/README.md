# swift-jinja root-cause fixes — HISTORICAL / ARCHIVAL

> **📜 ARCHIVAL — the fork is no longer pinned in `Package.swift`.**
>
> As of the swift-transformers 1.3.0 bump, vmlx-swift-lm transitively
> pulls `huggingface/swift-jinja` 2.3.5+, which contains all three
> root-cause fixes documented below (independently discovered by the
> upstream maintainers after our fork diverged). The
> `osaurus-ai/Jinja 1.3.1` fork at
> <https://github.com/osaurus-ai/Jinja> still exists as a reference
> implementation for anyone pinned to the 1.3.0 line, but vmlx no
> longer needs it.
>
> **What stays:** `ChatTemplateFallbacks.swift` + `TokenizerBridge`
> auto-engage. These sit above any Jinja version and catch future
> template regressions regardless of which upstream you're on.
>
> **Original writeup below kept verbatim** because the root-cause
> analysis is still useful context for anyone debugging the next
> Jinja regression.
>
> ---
>
> **Fork source:** https://github.com/osaurus-ai/Jinja
> **Branch:** `osaurus/1.3.0-patched`
> **Tag:** `1.3.1` (SemVer-bump on top of upstream `1.3.0` / commit
> `5c0a878`)
> **Identity match:** upstream URL = `https://github.com/johnmai-dev/Jinja`
> (last-path-component identity = `jinja`); our fork at
> `https://github.com/osaurus-ai/Jinja` has the SAME trailing
> `Jinja` so SPM treats it as the same package and overrides.
>
> **If you remove the fork pin from `Package.swift` or change its
> URL / tag, Gemma-4 and Nemotron-Cascade-2 chat templates will
> silently fall back to the hand-authored minimal templates in
> `ChatTemplateFallbacks.swift`.** The fallback still works but
> (a) ships a suboptimal prompt shape for Nemotron (uses Gemma-
> flavoured delimiters instead of the native
> `[AVAILABLE_TOOLS]` / `<|im_start|>` markers), (b) can't emit
> the full harmony channel envelope the native Gemma template
> uses for reasoning, and (c) adds a stderr warning every call
> when `VMLX_CHAT_TEMPLATE_FALLBACK_LOG=1`.

## Why not upstream?

1. `huggingface/swift-transformers` is pinned at `>= 0.1.21`, which
   transitively requires `jinja >= 1.3.0, < 1.4.0`. We can't jump
   to a newer Jinja major without swift-transformers updating first.
2. `johnmai-dev/swift-jinja` (the renamed upstream) HEAD already
   carries an unrelated **breaking API change** — `Template.render`
   takes `[String: Value]` instead of `[String: Any]`. Adopting HEAD
   requires a coordinated upgrade of every call site in
   swift-transformers AND any downstream consumer.
3. The three bugs are isolated to `Lexer.swift` and `Runtime.swift`
   with no API surface change. Forking `1.3.0` + patching +
   re-tagging as `1.3.1` is the minimum-disruption fix.

**Long-term plan:** file three upstream PRs against
`johnmai-dev/swift-jinja` HEAD; when upstream merges AND
swift-transformers bumps to a Jinja version that includes them,
drop this fork from `Package.swift`.

## The three bugs

All three were discovered during a multi-model chat-template audit
on 2026-04-23 (see `vmlx-swift-lm:main@7b95bca`'s investigation
follow-up). Each has a minimal Jinja-only reproducer that requires
no model download.

### 1. Lexer — `{{%` and `{{{` ambiguity

**File:** `Sources/Lexer.swift`, text-mode exit in the `main:` loop.
**Affects:** Gemma-4 (26B-A4B-it-*, E2B/E4B, 31B-JANG_4M).
**Pre-patch error:** `JinjaError.syntax("Unexpected token: multiplicativeBinaryOperator")`.

#### Root cause

Two preprocess regex substitutions trim whitespace around Jinja
delimiters:

```swift
template.replacing(#/\s*{%-/#, with: "{%")
template.replacing(#/\s*{{-/#, with: "{{")
```

When a raw template has a literal `{` at the end of a text region
immediately followed by whitespace and then `{%-` (or `{{-`) on the
next line — this is what Gemma-4 writes on line 83-85:

```
,parameters:{
{%- if params['properties'] -%}
    properties:{ {{- format_parameters(...) -}} },
{%- endif -%}
```

…preprocess eats the whitespace and yields `,parameters:{{%` (or
`{{{` for the `{{-` case). The lexer's text-mode exit check then
matches `{{` greedily as if it were an expression-opener, even
though the first `{` was a literal character. Inside expression
mode, the following `%` is matched as
`.multiplicativeBinaryOperator` (triggering the Gemma-4 error) or
the following `{` is matched as `.openCurlyBracket` (an object-
literal start, producing the second error we found during
debugging: `"Expected colon between key and value in object
literal. closeExpression != colon."`).

#### Fix

Extend the text-mode exit check to carve out the two ambiguous
triples:

- `{{%` → literal `{` + statement-opener `{%`
- `{{{` → literal `{` + expression-opener `{{`

For both, consume the first `{` as text and continue; the next
iteration exits text mode cleanly on the real delimiter.

See `0001-lexer-curly-ambiguity.patch`.

#### Minimal reproducer

```jinja
{%- for k in [1] -%}{{ k }}:{{%- endfor -%}
```

Expected: `1:{`. Pre-patch: syntax error at the `%` after `{{`.

### 2. Runtime — dict iteration binds single-identifier loopvar to `ArrayValue`

**File:** `Sources/Runtime.swift`, `evaluateFor` ObjectValue branch.
**Affects:** Nemotron-Cascade-2 30B-A3B family.
**Pre-patch error:** `JinjaError.runtime("Unknown operation type: not in")`.

#### Root cause

```swift
} else if let objectIterable = iterable as? ObjectValue {
    for (key, value) in objectIterable {
        let current = ArrayValue(value: [StringValue(value: key), value])
        …
        if let identifier = node.loopvar as? Identifier {
            scopeUpdateFunction = { scope in
                try scope.setVariable(name: identifier.value, value: current)
            }
        } else if let tupleLiteral = node.loopvar as? TupleLiteral {
            …
```

CPython Jinja2 semantics: `for k in dict` binds `k` to the **key
(string)**; `for k, v in dict` unpacks the `(k, v)` tuple. swift-
jinja 1.3.0 always binds the `ArrayValue([key, value])` tuple —
even when the loop variable is a single identifier. Downstream
expressions like `k in handled_keys` (where `handled_keys` is a
list of strings) then hit the `ArrayValue × ArrayValue` containment
branch, which only implements `+` and throws on everything else.

Triggers on Nemotron's `render_extra_keys(json_dict, handled_keys)`
macro:

```jinja
{%- for json_key in json_dict if json_key not in handled_keys %}
```

#### Fix

In the single-identifier branch, bind directly to
`StringValue(value: key)`. The tuple-literal branch is unchanged.

See `0002-runtime-dict-iter-and-select-expression.patch`, first hunk.

#### Minimal reproducer

```jinja
{%- macro R(d, h) -%}
    {%- for k in d if k not in h -%}{{ k }};{%- endfor -%}
{%- endmacro -%}
{{- R({'name':'x','age':1,'tool':'y'}, ['name','age']) -}}
```

Expected: `tool;`. Pre-patch: `Unknown operation type: not in`.

### 3. Runtime — standalone `SelectExpression` has no handler

**File:** `Sources/Runtime.swift`, top-level `evaluate` dispatch switch.
**Affects:** Gemma-4 with tools.
**Pre-patch error:** `JinjaError.runtime("Unknown node type: SelectExpression, …")`.

#### Root cause

`Parser.parseSelectExpression` produces a `SelectExpression` node
for the inline conditional `X if Y` (without `else`). The runtime
handles this node ONLY when it appears as a `for`-loop iterable
filter (`evaluateFor` line 356-358). The main `evaluate` switch
has no case for it — so `{{ ',' if not loop.last }}` (a valid
Jinja expression used in Gemma-4's tools loop tail) raises
`Unknown node type: SelectExpression`.

#### Fix

Add a case to the main `evaluate` switch: evaluate `test`; if
truthy, evaluate & return `iterable`; otherwise return
`UndefinedValue` (renders as empty string, matching CPython).

See `0002-runtime-dict-iter-and-select-expression.patch`, second hunk.

**Note:** Upstream `johnmai-dev/swift-jinja` HEAD already fixes
this bug at `Runtime.swift:1201`. This patch carries the same fix
back to the `1.3.0` branch.

#### Minimal reproducer

```jinja
{%- for item in [1, 2, 3] -%}
    {{ item }}{{ ',' if not loop.last }}
{%- endfor -%}
```

Expected: `1,2,3`. Pre-patch: `Unknown node type: SelectExpression`.

## Verification

Real-model matrix on M4 Max, Qwen 3.6 27B / 35B / Gemma-4 26B /
Nemotron-Cascade-2 30B — all 10/10 PASS via **NATIVE templates**
(no `VMLX_CHAT_TEMPLATE_FALLBACK_LOG` lines emitted). See
`vmlx-swift-lm:main`'s commit message for metrics.

Unit tests verifying each patch in isolation (no model needed) are
in the `Tests/MLXLMTests/` suite — search for the probe test
names: `testGemma4FullParse`, `testGemma4FullRenderWithTools`,
`testNemotronFullRender`, `testRegression*`.

## Regressions verified

The carve-outs in Lexer/Runtime are strictly additive — legitimate
Jinja constructs still parse/render correctly:

- `{{ a }}{{ b }}` — adjacent outputs
- `{% for n in [1,2] %}{{ n }},{% endfor %}` — simple loop
- `{% set d = {'a': 1, 'b': 2} %}{{ d['a'] }}` — object literal
- `for k, v in dict` — tuple unpacking (unchanged code path)
- `for arr in [[1,2],[3,4]]` — legitimate array binding
