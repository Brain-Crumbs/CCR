/** Per-tick predicted/assumed-vs-actuated motor action model, built from
 * DecisionRecord.motor_decision (motor/reflexes.py's MotorDecision):
 * `voluntary` is the action the organism intended (what it "assumed" it
 * would do); `actuated` is what it actually did once a reflex or a
 * caregiver override may have intervened. `decision.reflex` is a separate
 * orienting reflex (core/attention.py), not the motor reflex stack. */

function tickOf(decision, fallback) {
  return Number(decision?.tick_index ?? decision?.tick ?? fallback);
}

export function actionTimeline(decisions = []) {
  return decisions.map((decision, index) => {
    const motor = decision.motor_decision ?? null;
    const voluntary = motor?.voluntary ?? null;
    const actuated = motor?.actuated ?? voluntary;
    return {
      tick: tickOf(decision, index),
      voluntary,
      actuated,
      diverged: voluntary !== null && actuated !== null && voluntary !== actuated,
      motorReflex: motor?.reflex ?? null,
      caregiverOverride: motor?.caregiver_override ?? null,
      orienting: decision.reflex ?? null,
      arbiterMode: decision.arbiter_mode?.mode ?? decision.arbiter_mode?.value ?? null,
      predictionError: typeof decision.prediction_error === "number" ? decision.prediction_error : null,
    };
  });
}
