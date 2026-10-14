"""Pick the traces notebooks 3 and 6 work on, from notebook 2's assessments.

Both notebooks use the same rule, so the hand-made fix (nb3) and the automatic
one (nb6, GEPA) are measured on the same questions:
- issue set: traces that fail `sensitive_data_masking` or `conciseness`;
- regression set: the remaining traces that pass `accuracy` and `relevance`.
"""

import json

ISSUE_SCORERS = ('sensitive_data_masking', 'conciseness')
REGRESSION_SCORERS = ('accuracy', 'relevance')


def _row_assessment_dicts(a):
  """Assessment as a dict with 'assessment_name' / 'feedback' keys, whether the
  DataFrame stores dicts (list column) or Assessment entities (info column)."""
  if isinstance(a, dict):
    return a
  if hasattr(a, 'to_dictionary'):
    return a.to_dictionary()
  return {'assessment_name': getattr(a, 'name', None),
          'feedback': {'value': getattr(getattr(a, 'feedback', None), 'value', None)}}


def _question_text(request):
  """The user's question from a trace request. The traced AGENT.predict call
  records its arguments as JSON ({"request": {"input": [{"content": ...}]}}),
  so pull out the last user message instead of using the whole JSON blob."""
  try:
    data = json.loads(request) if isinstance(request, str) else request
    content = data['request']['input'][-1]['content']
    if isinstance(content, list):
      content = ' '.join(p.get('text', '') for p in content if isinstance(p, dict))
    return content
  except (TypeError, ValueError, KeyError, IndexError):
    return request


def _row_parts(trace_row):
  """Version-proof access to (assessments, trace_id, question) from one
  search_traces row - mlflow versions expose either an `info` column with a
  TraceInfo object or flat `assessments`/`trace_id`/`request` columns."""
  info = trace_row.get('info')
  if info is not None:
    raw = getattr(info, 'assessments', None) or []
    trace_id = getattr(info, 'trace_id', None) or trace_row.get('trace_id')
    question = trace_row.get('request') or getattr(info, 'request_preview', None)
  else:
    raw = trace_row.get('assessments') or []
    trace_id = trace_row.get('trace_id')
    question = trace_row.get('request')
  return [_row_assessment_dicts(a) for a in raw], trace_id, _question_text(question)


def select_issue_and_regression(traces, max_issues=10, max_regression=5):
  """Split a search_traces DataFrame into (issue_records, regression_records, issues).

  Records are eval-dataset rows: {'inputs': {'question': ...}, 'trace_id': ...}
  (inputs keyed by the predict_fn's parameter name). The sets are disjoint:
  the first `max_issues` traces with an issue go to the issue set; the
  remaining accurate + relevant traces fill the regression set. `issues` maps
  each issue trace_id to the scorers it failed.
  """
  issue_records, regression_records, issues = [], [], {}
  for _, trace_row in traces.iterrows():
    assessments, trace_id, question = _row_parts(trace_row)
    if not trace_id or not assessments:
      continue
    verdicts = {a.get('assessment_name'): (a.get('feedback') or {}).get('value')
                for a in assessments}
    failed = [name for name in ISSUE_SCORERS if verdicts.get(name) == 'no']
    record = {'inputs': {'question': question}, 'trace_id': trace_id}
    if failed and len(issue_records) < max_issues:
      issue_records.append(record)
      issues[trace_id] = failed
    elif (all(verdicts.get(name) == 'yes' for name in REGRESSION_SCORERS)
          and len(regression_records) < max_regression):
      regression_records.append(record)
  return issue_records, regression_records, issues
