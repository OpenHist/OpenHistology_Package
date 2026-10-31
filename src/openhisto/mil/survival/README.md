# Survival heads

`create_survival_head("cox", input_dim, network="linear"|"mlp", ...)` returns
scalar log-relative risks. `create_survival_head("discrete_hazard", input_dim,
network=..., num_bins=20)` returns one conditional-hazard logit per interval.
Both consume CPU bag-level embeddings `[N,D]` and return the shared
`openhisto.mil.heads.types.SurvivalOutput`. They never apply a classification softmax.
MLPs accept `hidden_dims`, `activation`, and `dropout`.

Use `event=1` for an observed event and `event=0` for right censoring. Durations
must be finite, nonnegative, and use one unchanged time unit throughout.

## Cox

`cox_ph_loss(log_risk, durations, events, ties="efron")` computes grouped risk
sets `R(t)={j:duration[j]>=t}`. Same-time censored observations remain at risk.
`ties="breslow"` is also supported. Mean loss is per observed event. A single
subject or no-event risk set is rejected by default. These are cohort losses:
independent one-slide Cox losses do not provide useful risk comparisons.

After fitting the head, call `head.fit_baseline(training_embeddings, durations,
events)` or `fit_cox_baseline(training_log_risks, durations, events)` followed by
`head.set_baseline(...)`. Calibration is explicitly training-only. Before this,
forward exposes risk and logits, with no absolute survival curve. Afterwards,
forward returns Breslow-calibrated cumulative hazards and survival. Refit the
baseline after changing model weights. Optional prediction `times` must be in
the fitted horizon; `allow_extrapolation=True` explicitly carries the final
baseline beyond that horizon.

## Discrete hazards

Call `head.fit_time_grid(training_durations, events)` before interpreting output
at times or computing loss. The default fits 20 equal-width intervals with 21
boundaries from zero through maximum training follow-up. Alternatively supply
explicit `boundaries=[0,...,T]` when constructing the head. Boundaries and their
effective interval count are saved. Validation/test observations never expand
the fitted grid.

Intervals are `[0,b1], (b1,b2], ...`. An event contributes survival through its
preceding intervals and its event-bin hazard. A censor contributes only fully
completed intervals; censoring before the first right endpoint contributes no
term. Events/censors exactly at a boundary belong to the preceding interval.

`head.loss(output, durations, events)` evaluates this likelihood. Values beyond
the fitted horizon raise by default. Explicit `administrative_truncation=True`
treats them as censored at the horizon. `return_targets=True` returns the loss
and `DiscreteTargets`, which keeps original observations and truncation counts;
subject labels are never saved in the head's state/config.

Discrete `risk=-sum(bin_survival)` follows the existing native MIL convention;
it is not expected survival time. Before grid fitting, time-indexed `survival`
and `time_grid` are absent; unindexed `bin_survival` is explicitly marked in aux.

## Persistence and checks

Use `openhisto.mil.heads.save_head/load_head` for the shared tensor-only artifact format.
Configs recreate linear/MLP architecture and fitted boundaries. Aggregate Cox
baseline buffers live in log space with a risk-centering offset and resize on
strict load; individual training observations are not stored.

Run synthetic checks with:

```sh
python -m unittest discover -s src/openhisto/mil/survival/tests -t src -v
```

The tests cover hand likelihood and Breslow oracles, tied-event permutations,
same-time censoring, interval boundaries, administrative truncation, extreme
values, gradients, and calibrated artifact roundtrips.
