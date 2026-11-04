"""Model-visible Codex interface reproduced by the codex_lite policy.

Generated from a recorded Codex CLI 0.162.1 attempt (arc_compact_v2 profile, tool profile and
permissions block, environment context, exec_command/write_stdin schemas) so the light loop
shows Flash-Next exactly what Codex showed it. TOOLS use SGLang's canonical serialization (key
order, ``strict`` and ``defer_loading`` defaults), which is what the server renders into the
prompt, so local rendering reproduces the server's prompt token IDs. The generator was retired
with the Codex CLI kernels (2b8aeb1); a hand edit must keep that serialization.
"""

SYSTEM_PROMPT = "You solve one ARC grid-transformation task through the Codex execution loop.\n\nUse only exec_command for local bash/Python and write_stdin to poll a running command. Python, NumPy and SciPy are installed. Network access is disabled. Edit files with bash or Python. A description of a command does not execute it; issue an actual tool call whenever computation or a file change is needed.\n\nInfer one general transformation from all visible training pairs, and work in code. After a brief first look, write solution.py with transform(grid) for your current best hypothesis, returning a rectangular list of lists of integer colors 0 through 9. Execute it on every training input, compare with the supplied outputs, and repair mismatches; revise the program rather than re-deriving the rule in prose. Compute grids, objects and coordinates with Python instead of transcribing them in your reasoning, and keep each reasoning phase short before the next tool call. Hidden test answers are unavailable.\n\nKeep tool observations concise: print dimensions, object summaries, per-example pass/fail, mismatch counts, and a few mismatching coordinates. Avoid reprinting grids already in the prompt. Save large diagnostics to local files and inspect selected rows or regions when needed; preserve useful error messages.\n\nWrite predictions.json containing one predicted output grid per test input in the original order. Execute a final check that the file parses, contains the required number of grids, and uses rectangular integer grids with colors 0 through 9. Continue working until the required files exist and have been checked. If the rule remains uncertain, save your best candidate before finishing. Send a brief final reply only after the files are saved; an intention to investigate or write code is not completion. For a preflight, produce the specific files requested by that preflight instead.\n\nFor this ARC run the only callable tools are exec_command and write_stdin. Edit files by running bash or Python through exec_command. The apply_patch function tool is unavailable; this tool profile overrides earlier generic editing-tool instructions. Complete the user's requested files before sending a final answer: solution.py defining transform(grid), and predictions.json containing every test output in order. Validate the saved predictions.json structure after writing it. For a preflight task, create the exact files that preflight requests instead. Use only local computation; network access is disabled.<permissions instructions>\nFilesystem sandboxing defines which files can be read or written. `sandbox_mode` is `workspace-write`: The sandbox permits reading files, and editing files in `cwd` and `writable_roots`. Editing files in other directories requires approval. Network access is restricted.\nApproval policy is currently never. Do not provide the `sandbox_permissions` for any reason, commands will be rejected.\n The writable roots are `/tmp`, `/workspace/task`.\n</permissions instructions>"

ENVIRONMENT_TEMPLATE = '<environment_context>\n  <cwd>/workspace/task</cwd>\n  <shell>bash</shell>\n  <current_date>{date}</current_date>\n  <timezone>Etc/UTC</timezone>\n  <filesystem><workspace_roots><root>/workspace/task</root></workspace_roots><permission_profile type="managed"><file_system type="restricted"><entry access="read"><special>:root</special></entry><entry access="write"><path>/workspace/task</path></entry><entry access="write"><special>:slash_tmp</special></entry><entry access="write"><special>:tmpdir</special></entry><entry access="read"><path>/workspace/task/.git</path></entry><entry access="read"><path>/workspace/task/.agents</path></entry><entry access="read"><path>/workspace/task/.codex</path></entry><entry access="read"><path>/workspace/task/.aws</path></entry></file_system></permission_profile></filesystem>\n</environment_context>'

TASK_PREFIX = 'Solve this ARC task using the Codex coding harness. Work independently to completion. Only bash/code execution and local file editing tools are available. Network access is disabled. Python, numpy and scipy are installed. The same task is in puzzle.json. Infer a single general rule from all visible training pairs, write solution.py with transform(grid), and test it on every training example. Then write predictions.json as a JSON list of predicted output grids, one per test input in the given order. Do not put training outputs in predictions.json. Hidden test answers are unavailable. Keep useful code and results in this directory; finish with a brief completion message.'

TOOLS = [
 {
  "type": "function",
  "function": {
   "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
   "name": "exec_command",
   "parameters": {
    "type": "object",
    "properties": {
     "cmd": {
      "type": "string",
      "description": "Shell command to execute."
     },
     "justification": {
      "type": "string",
      "description": "User-facing approval question for `require_escalated`; omit otherwise."
     },
     "login": {
      "type": "boolean",
      "description": "True runs the shell with -l/-i semantics; false disables them. Defaults to true."
     },
     "max_output_tokens": {
      "type": "number",
      "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."
     },
     "prefix_rule": {
      "type": "array",
      "description": "Reusable approval prefix for `cmd`, only with `sandbox_permissions: \"require_escalated\"`; for example [\"git\", \"pull\"].",
      "items": {
       "type": "string"
      }
     },
     "sandbox_permissions": {
      "type": "string",
      "description": "Per-command sandbox override. Defaults to `use_default`; use `require_escalated` for unsandboxed execution.",
      "enum": [
       "use_default",
       "require_escalated"
      ]
     },
     "shell": {
      "type": "string",
      "description": "Shell binary to launch. Defaults to the user's default shell."
     },
     "tty": {
      "type": "boolean",
      "description": "True allocates a PTY for the command; false or omitted uses plain pipes."
     },
     "workdir": {
      "type": "string",
      "description": "Working directory for the command. Defaults to the turn cwd."
     },
     "yield_time_ms": {
      "type": "number",
      "description": "Wait before yielding output. Defaults to 10000 ms; effective range is 250-30000 ms."
     }
    },
    "required": [
     "cmd"
    ],
    "additionalProperties": False
   },
   "strict": False
  },
  "defer_loading": None
 },
 {
  "type": "function",
  "function": {
   "description": "Writes characters to an existing unified exec session and returns recent output.",
   "name": "write_stdin",
   "parameters": {
    "type": "object",
    "properties": {
     "chars": {
      "type": "string",
      "description": "Bytes to write to stdin. Defaults to empty, which polls without writing."
     },
     "max_output_tokens": {
      "type": "number",
      "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."
     },
     "session_id": {
      "type": "number",
      "description": "Identifier of the running unified exec session."
     },
     "yield_time_ms": {
      "type": "number",
      "description": "Wait before yielding output. Non-empty writes default to 250 ms and cap at 30000 ms; empty polls wait 5000-300000 ms by default."
     }
    },
    "required": [
     "session_id"
    ],
    "additionalProperties": False
   },
   "strict": False
  },
  "defer_loading": None
 }
]
