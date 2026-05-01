// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Hard-coded fallback chat templates. Used by the huggingface tokenizer
// bridge when the model's native `chat_template.jinja` triggers a
// swift-jinja parser/runtime bug it can't evaluate. The bridge tries
// each candidate in order and uses the first one that renders;
// downstream consumers see no difference.
//
// Known upstream limitations these templates work around
// (johnmai-dev/Jinja 1.3.x):
//
//   - Gemma-4 templates (26B-A4B-it-*, E2B/E4B, 31B-JANG_4M):
//     `JinjaError.syntax("Unexpected token: multiplicativeBinaryOperator")`
//     at parse. Individual constructs parse fine but the full template
//     assembly trips the parser.
//
//   - Nemotron-Cascade-2 templates: `JinjaError.runtime("Unknown
//     operation type: not in")`. Root cause is in swift-jinja's for-
//     loop runtime — when iterating a dict with `for k in d`, the loop
//     var is bound to an `ArrayValue([key, value])` rather than the
//     scalar key, and the `not in` containment check hits the
//     ArrayValue × ArrayValue branch which only handles `+`.
//
// These fallbacks are intentionally minimal — they keep the prompt
// contract (role markers, tool declaration, generation-prompt suffix)
// but drop the complex formatting (BNF-style parameter blocks, etc.)
// that's either upstream-bug territory or purely cosmetic.

import Foundation

public enum ChatTemplateFallbacks {

    /// Gemma-4 text-only + image / video / audio, no tools. Preserves
    /// `<|turn>role` / `<turn|>` delimiters that the Gemma-4 model
    /// family was trained on.
    public static let gemma4Minimal: String = #"""
{{- bos_token -}}
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content | trim -}}
    {%- elif content is sequence -%}
        {%- for item in content -%}
            {%- if item['type'] == 'text' -%}
                {{- item['text'] | trim -}}
            {%- elif item['type'] == 'image' -%}
                {{- '\n\n<|image|>\n\n' -}}
            {%- elif item['type'] == 'video' -%}
                {{- '\n\n<|video|>\n\n' -}}
            {%- elif item['type'] == 'audio' -%}
                {{- '\n\n<|audio|>\n\n' -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- if messages[0]['role'] == 'system' -%}
    {{- '<|turn>system\n' -}}
    {{- render_content(messages[0]['content']) -}}
    {{- '<turn|>\n' -}}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
    {{- '<|turn>' + role + '\n' -}}
    {{- render_content(message['content']) -}}
    {{- '<turn|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|turn>model\n' -}}
{%- endif -%}
"""#

    /// Gemma-4 with a `<|tool>...<tool|>` declaration block for each
    /// tool and `<|tool_call>call:name{args}<tool_call|>` assistant
    /// output. Tool replies render as
    /// `<|tool_response>response:name{content}<tool_response|>`.
    public static let gemma4WithTools: String = #"""
{{- bos_token -}}
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content | trim -}}
    {%- elif content is sequence -%}
        {%- for item in content -%}
            {%- if item['type'] == 'text' -%}
                {{- item['text'] | trim -}}
            {%- elif item['type'] == 'image' -%}
                {{- '\n\n<|image|>\n\n' -}}
            {%- elif item['type'] == 'video' -%}
                {{- '\n\n<|video|>\n\n' -}}
            {%- elif item['type'] == 'audio' -%}
                {{- '\n\n<|audio|>\n\n' -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- macro render_tool_args(arguments) -%}
    {%- if arguments is mapping -%}
        {%- set first = true -%}
        {%- for key, value in arguments | dictsort -%}
            {%- if not first %},{% endif -%}
            {%- set first = false -%}
            {{- key -}}:{%- if value is string -%}<|"|>{{ value }}<|"|>
                {%- elif value is boolean -%}{{ 'true' if value else 'false' }}
                {%- else -%}{{ value }}
            {%- endif -%}
        {%- endfor -%}
    {%- elif arguments is string -%}
        {{- arguments -}}
    {%- endif -%}
{%- endmacro -%}
{%- if (tools or (messages[0]['role'] in ['system', 'developer'])) -%}
    {{- '<|turn>system\n' -}}
    {%- if messages[0]['role'] in ['system', 'developer'] -%}
        {{- render_content(messages[0]['content']) -}}
        {%- set loop_messages = messages[1:] -%}
    {%- else -%}
        {%- set loop_messages = messages -%}
    {%- endif -%}
    {%- if tools -%}
        {%- for tool in tools -%}
            {{- '<|tool>declaration:' + tool['function']['name'] -}}
            {%- if tool['function']['description'] -%}
                {{- '{description:<|"|>' + tool['function']['description'] + '<|"|>}' -}}
            {%- endif -%}
            {{- '<tool|>' -}}
        {%- endfor -%}
    {%- endif -%}
    {{- '<turn|>\n' -}}
{%- else -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
    {%- if message['role'] == 'tool' -%}
        {{- '<|tool_response>response:' + (message.get('name') or 'unknown') + '{' -}}
        {{- render_content(message['content']) -}}
        {{- '}<tool_response|>\n' -}}
    {%- else -%}
        {{- '<|turn>' + role + '\n' -}}
        {%- if message['content'] -%}
            {{- render_content(message['content']) -}}
        {%- endif -%}
        {%- if message['tool_calls'] -%}
            {%- for tool_call in message['tool_calls'] -%}
                {%- set fn = tool_call['function'] -%}
                {{- '<|tool_call>call:' + fn['name'] + '{' -}}
                {{- render_tool_args(fn['arguments']) -}}
                {{- '}<tool_call|>' -}}
            {%- endfor -%}
        {%- endif -%}
        {{- '<turn|>\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|turn>model\n' -}}
{%- endif -%}
"""#

    /// Nemotron-Cascade-2 minimal. Avoids the `for k in dict if k not
    /// in handled_keys` construct that trips swift-jinja's runtime.
    /// Uses the ChatML-style `<|im_start|>role` / `<|im_end|>` turn
    /// markers Nemotron was actually trained on (the first attempt
    /// incorrectly used `<extra_id_*>`; see tokenizer special-token
    /// inspection — `<|im_start|>` + `[AVAILABLE_TOOLS]` are the real
    /// markers). Tool declarations use the `[AVAILABLE_TOOLS]` /
    /// `[/AVAILABLE_TOOLS]` block and assistant tool calls use the
    /// `<tool_call><function=name></function></tool_call>` XML form.
    public static let nemotronMinimal: String = #"""
{%- set loop_messages = messages -%}
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = 'You are a helpful and harmless assistant.' -%}
{%- endif -%}
<|im_start|>system
{{ system_message }}
{%- if tools %}

[AVAILABLE_TOOLS]
{%- for tool in tools %}
  {%- set fn = tool['function'] if tool['function'] is defined else tool -%}
  <tool>
    <name>{{ fn['name'] }}</name>
    {%- if fn['description'] is defined %}
    <description>{{ fn['description'] | trim }}</description>
    {%- endif %}
    {%- if fn['parameters'] is defined and fn['parameters']['properties'] is defined %}
    <parameters>
      {%- for param_name, param in fn['parameters']['properties'] | dictsort %}
      <parameter>
        <name>{{ param_name }}</name>
        {%- if param['type'] is defined %}<type>{{ param['type'] }}</type>{%- endif %}
        {%- if param['description'] is defined %}<description>{{ param['description'] | trim }}</description>{%- endif %}
      </parameter>
      {%- endfor %}
    </parameters>
    {%- endif %}
  </tool>
{%- endfor %}
[/AVAILABLE_TOOLS]
{%- endif %}
<|im_end|>
{% for message in loop_messages -%}
{%- if message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}
<|im_end|>
{%- elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{%- if message['content'] -%}
{{ message['content'] }}
{%- endif %}
{%- if message['tool_calls'] is defined and message['tool_calls'] %}
{%- for tc in message['tool_calls'] %}
<tool_call>
<function={{ tc['function']['name'] }}>
{%- if tc['function']['arguments'] is mapping %}
{%- for k, v in tc['function']['arguments'] | dictsort %}
<parameter={{ k }}>
{{ v }}
</parameter>
{%- endfor %}
{%- elif tc['function']['arguments'] is string -%}
{{ tc['function']['arguments'] }}
{%- endif %}
</function>
</tool_call>
{%- endfor %}
{%- endif %}
<|im_end|>
{%- elif message['role'] == 'tool' -%}
<|im_start|>tool
{{ message['content'] }}
<|im_end|>
{%- endif -%}

{% endfor -%}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}
"""#

    /// DeepSeek-V4 minimal template. DSV4-Flash bundles ship NO
    /// `chat_template` field in `tokenizer_config.json` — the stock
    /// distribution carries an external `encoding/encoding_dsv4.py`
    /// instead. This jinja renders the same wire format the Python
    /// encoder produces (BOS / `<｜User｜>` / `<｜Assistant｜>` /
    /// closed `</think>` chat-mode tail / open `<think>` thinking-
    /// mode tail / DSML tool calls / `reasoning_effort=max` preface).
    /// Selected via the DSV4 BOS sniff in the tokenizer bridge.
    public static let dsv4Minimal: String = #"""
{%- set bos = '<｜begin▁of▁sentence｜>' -%}
{%- set eos = '<｜end▁of▁sentence｜>' -%}
{%- set user_token = '<｜User｜>' -%}
{%- set asst_token = '<｜Assistant｜>' -%}
{%- set think_open = '<think>' -%}
{%- set think_close = '</think>' -%}
{%- set dsml = '｜DSML｜' -%}
{{- bos -}}
{%- if reasoning_effort == 'max' -%}
{{- 'Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n' -}}
{%- endif -%}
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{- message['content'] -}}
{%- elif message['role'] == 'user' or message['role'] == 'developer' -%}
{{- user_token -}}{{- message['content'] -}}
{%- elif message['role'] == 'assistant' -%}
{{- asst_token -}}
{%- if message.get('reasoning_content') -%}
{{- message['reasoning_content'] -}}{{- think_close -}}
{%- endif -%}
{{- message['content'] or '' -}}
{{- eos -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- asst_token -}}
{%- if enable_thinking -%}
{{- think_open -}}
{%- else -%}
{{- think_close -}}
{%- endif -%}
{%- endif -%}
"""#

    /// REMOVED 2026-05-01 — kept as `_lagunaMinimal_DEPRECATED` for one
    /// release in case the osaurus-ai/swift-jinja parseOr fork breaks
    /// on a Laguna template variant we haven't tested. Verified that
    /// the native `laguna_glm_thinking_v5/chat_template.jinja` (with
    /// `{%- generation -%}…{%- endgeneration -%}` block tags) parses
    /// AND renders correctly through the bridge against the swift-jinja
    /// fork at 58d21aa5. End-to-end Laguna BENCH_BATCH_CHAT 3-turn is
    /// coherent ("That's a great color!..." → "Your favorite color is
    /// blue!..." → "Blue is a cool color..."). No fallback engaged in
    /// the verified path; the bridge sniff that previously force-routed
    /// to this template was removed in the same commit.
    private static let _lagunaMinimal_DEPRECATED: String = #"""
{{- "〈|EOS|〉" -}}
{%- set enable_thinking = enable_thinking | default(false) -%}
{%- set add_generation_prompt = add_generation_prompt | default(false) -%}
{%- set system_message = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations." -%}
{%- if messages and messages[0].role == "system" -%}
  {%- set system_message = messages[0].content -%}
{%- endif -%}
{%- if (system_message and system_message.strip()) or tools -%}
  {{- "<system>\n" -}}
  {%- if system_message and system_message.strip() -%}
    {{- "\n" -}}{{- system_message.rstrip() -}}
  {%- endif -%}
  {%- if tools -%}
    {{- "\n\n### Tools\n\n" -}}
    {{- "You may call functions to assist with the user query.\n" -}}
    {{- "All available function signatures are listed below:\n" -}}
    {{- "<available_tools>\n" -}}
    {%- for tool in tools -%}
      {{- tool | tojson -}}{{- "\n" -}}
    {%- endfor -%}
    {{- "</available_tools>\n" -}}
  {%- endif -%}
  {{- "\n</system>\n" -}}
{%- endif -%}
{%- for message in messages -%}
  {%- if message.role == "system" -%}
    {#- handled above -#}
  {%- elif message.role == "user" -%}
    {{- "<user>\n" -}}
    {%- if message.content is string -%}
      {{- message.content -}}
    {%- else -%}
      {%- for item in message.content -%}
        {%- if item.type == "text" -%}{{- item.text -}}{%- endif -%}
      {%- endfor -%}
    {%- endif -%}
    {{- "\n</user>\n" -}}
  {%- elif message.role == "assistant" -%}
    {{- "<assistant>\n" -}}
    {%- if enable_thinking -%}
      {{- "<think>\n" -}}
    {%- else -%}
      {{- "</think>\n" -}}
    {%- endif -%}
    {%- if message.content is string -%}
      {{- message.content -}}
    {%- else -%}
      {%- for item in message.content -%}
        {%- if item.type == "text" -%}{{- item.text -}}{%- endif -%}
      {%- endfor -%}
    {%- endif -%}
    {{- "\n</assistant>\n" -}}
  {%- elif message.role == "tool" -%}
    {{- "<tool_response>\n" -}}
    {%- if message.content is string -%}
      {{- message.content -}}
    {%- else -%}
      {{- message.content | tojson -}}
    {%- endif -%}
    {{- "\n</tool_response>\n" -}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- "<assistant>\n" -}}
  {%- if enable_thinking -%}
    {{- "<think>\n" -}}
  {%- else -%}
    {{- "</think>\n" -}}
  {%- endif -%}
{%- endif -%}
"""#

    /// Mistral 3 / Mistral 3.5 / Mistral-Medium-3.5 minimal fallback.
    ///
    /// The native template uses
    /// `{%- for message in loop_messages + [{'role':'__sentinel__'}] %}`
    /// — swift-jinja can't parse `+` concatenation inside the
    /// for-iterable expression, throwing `Expected '%}' after for
    /// loop.. Got plus instead`. Reported on real
    /// `Mistral-Medium-3.5-128B-JANGTQ` 2026-05-01.
    ///
    /// This fallback preserves the Mistral-family markers
    /// `[SYSTEM_PROMPT] / [INST] / [/INST] / [AVAILABLE_TOOLS] /
    /// [/AVAILABLE_TOOLS] / [TOOL_CALLS] / [TOOL_RESULTS]`. Drops
    /// the consecutive-message aggregation logic the native template
    /// uses (which is what needed the `+ [sentinel]` trick) — most
    /// chats don't have consecutive same-role messages.
    ///
    /// Bridge sniff: `convertTokenToId("[INST]") != nil` is the
    /// cleanest Mistral-family signal — it's a special token across
    /// every Mistral 3 / 3.5 / 4 distribution.
    /// REMOVED 2026-05-01 — same rationale as the Laguna deprecation
    /// directly above. The osaurus-ai/swift-jinja fork (58d21aa5)
    /// fixes the for-loop-iterable parser to accept binary `+`
    /// expressions, so Mistral 3.5's native template
    /// (`{%- for message in loop_messages + [{...}] %}`) now parses
    /// and renders correctly with the FULL `[MODEL_SETTINGS]
    /// {"reasoning_effort": "..."}` plumbing the model was trained on.
    /// Kept private as `_mistral3Minimal_DEPRECATED` for one release
    /// as defensive reference.
    private static let _mistral3Minimal_DEPRECATED: String = #"""
{%- if messages and messages[0].role == "system" -%}
  {{- "[SYSTEM_PROMPT]" -}}
  {%- if messages[0].content is string -%}
    {{- messages[0].content -}}
  {%- else -%}
    {%- for item in messages[0].content -%}
      {%- if item.type == "text" -%}{{- item.text -}}{%- endif -%}
    {%- endfor -%}
  {%- endif -%}
  {{- "[/SYSTEM_PROMPT]" -}}
{%- endif -%}
{%- set effort = reasoning_effort | default("none") -%}
{{- "[MODEL_SETTINGS]" -}}
{{- '{"reasoning_effort": "' -}}{{- effort -}}{{- '"}' -}}
{{- "[/MODEL_SETTINGS]" -}}
{%- if tools -%}
  {{- "[AVAILABLE_TOOLS]" -}}
  {{- tools | tojson -}}
  {{- "[/AVAILABLE_TOOLS]" -}}
{%- endif -%}
{%- for message in messages -%}
  {%- if message.role == "system" -%}
    {#- handled above -#}
  {%- elif message.role == "user" -%}
    {{- "[INST]" -}}
    {%- if message.content is string -%}
      {{- message.content -}}
    {%- else -%}
      {%- for item in message.content -%}
        {%- if item.type == "text" -%}{{- item.text -}}{%- endif -%}
        {%- if item.type == "image" -%}{{- "[IMG]" -}}{%- endif -%}
      {%- endfor -%}
    {%- endif -%}
    {{- "[/INST]" -}}
  {%- elif message.role == "assistant" -%}
    {%- if message.tool_calls -%}
      {{- "[TOOL_CALLS]" -}}
      {{- message.tool_calls | tojson -}}
      {{- "[/TOOL_CALLS]" -}}
    {%- else -%}
      {%- if message.content is string -%}
        {{- message.content -}}
      {%- else -%}
        {%- for item in message.content -%}
          {%- if item.type == "text" -%}{{- item.text -}}{%- endif -%}
        {%- endfor -%}
      {%- endif -%}
      {{- eos_token -}}
    {%- endif -%}
  {%- elif message.role == "tool" -%}
    {{- "[TOOL_RESULTS]" -}}
    {%- if message.content is string -%}
      {{- message.content -}}
    {%- else -%}
      {{- message.content | tojson -}}
    {%- endif -%}
    {{- "[/TOOL_RESULTS]" -}}
  {%- endif -%}
{%- endfor -%}
"""#

    /// Ordered list of (label, template) fallbacks used when the
    /// model's native template throws. Order matters: `gemma4WithTools`
    /// comes first because (a) it subsumes `gemma4Minimal` when no
    /// tools are present, and (b) Gemma-4 is the most common family
    /// blocked by the upstream parser bug.
    public static let orderedFallbacks: [(label: String, template: String)] = [
        ("Gemma4WithTools", gemma4WithTools),
        ("Gemma4Minimal",   gemma4Minimal),
        ("NemotronMinimal", nemotronMinimal),
        ("DSV4Minimal",     dsv4Minimal),
    ]
}
