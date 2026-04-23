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

    /// Ordered list of (label, template) fallbacks used when the
    /// model's native template throws. Order matters: `gemma4WithTools`
    /// comes first because (a) it subsumes `gemma4Minimal` when no
    /// tools are present, and (b) Gemma-4 is the most common family
    /// blocked by the upstream parser bug.
    public static let orderedFallbacks: [(label: String, template: String)] = [
        ("Gemma4WithTools", gemma4WithTools),
        ("Gemma4Minimal",   gemma4Minimal),
        ("NemotronMinimal", nemotronMinimal),
    ]
}
