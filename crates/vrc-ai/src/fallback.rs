//! Local rule-based autonomous decision fallback when remote/local AI models are unavailable.

use vrc_schema::{DecisionAction, DecisionContext, DecisionOutput, EmotionTag};

const MOOD_CYCLE: &[&str] = &["happy", "shy", "surprised", "neutral"];

/// Evaluates a context-aware autonomous mascot action locally without any LLM network calls.
pub fn local_fallback_decision(
    ctx: &DecisionContext,
    window_perch_enabled: bool,
    is_seated_on_window: bool,
    tick: u64,
) -> DecisionOutput {
    if ctx.pet_count_recent > 0 && (tick.is_multiple_of(2) || ctx.current_mood != EmotionTag::Happy)
    {
        return DecisionOutput {
            action: DecisionAction::ChangeExpression,
            target_expression: Some("happy".to_owned()),
            speech_intent: None,
            reason: Some("[ローカル自律] 撫でられてご機嫌です".to_owned()),
        };
    }

    if (ctx.time_of_day == "night" || ctx.idle_seconds > 180.0) && tick % 4 == 1 {
        return DecisionOutput {
            action: DecisionAction::ChangeExpression,
            target_expression: Some("sleepy".to_owned()),
            speech_intent: None,
            reason: Some("[ローカル自律] 少し眠くなってきました".to_owned()),
        };
    }

    if is_seated_on_window && tick % 5 == 4 {
        return DecisionOutput {
            action: DecisionAction::MoveToTaskbar,
            target_expression: Some("neutral".to_owned()),
            speech_intent: None,
            reason: Some("[ローカル自律] タスクバーに戻って休憩します".to_owned()),
        };
    }

    if window_perch_enabled && !is_seated_on_window && tick % 5 == 2 {
        return DecisionOutput {
            action: DecisionAction::SitOnWindow,
            target_expression: Some("happy".to_owned()),
            speech_intent: None,
            reason: Some("[ローカル自律] ウィンドウの上に移動してみます".to_owned()),
        };
    }

    if tick % 4 == 3 {
        let idx = ((tick / 4) as usize) % MOOD_CYCLE.len();
        let expr = MOOD_CYCLE[idx];
        return DecisionOutput {
            action: DecisionAction::ChangeExpression,
            target_expression: Some(expr.to_owned()),
            speech_intent: None,
            reason: Some("[ローカル自律] 気分転換に表情を変えました".to_owned()),
        };
    }

    DecisionOutput {
        action: DecisionAction::Idle,
        target_expression: None,
        speech_intent: None,
        reason: Some("[ローカル自律] そばで静かに見守っています".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_reacts_to_petting_night_perch_and_mood_cycles() {
        let pet_ctx = DecisionContext {
            pet_count_recent: 2,
            ..Default::default()
        };
        let pet_out = local_fallback_decision(&pet_ctx, true, false, 0);
        assert_eq!(pet_out.action, DecisionAction::ChangeExpression);
        assert_eq!(pet_out.target_expression.as_deref(), Some("happy"));
        assert!(pet_out.reason.unwrap().starts_with("[ローカル自律]"));

        let base_ctx = DecisionContext {
            time_of_day: "afternoon".to_owned(),
            ..Default::default()
        };
        let perch_out = local_fallback_decision(&base_ctx, true, false, 2);
        assert_eq!(perch_out.action, DecisionAction::SitOnWindow);

        let return_out = local_fallback_decision(&base_ctx, true, true, 4);
        assert_eq!(return_out.action, DecisionAction::MoveToTaskbar);

        let night_ctx = DecisionContext {
            time_of_day: "night".to_owned(),
            ..Default::default()
        };
        let sleepy_out = local_fallback_decision(&night_ctx, false, false, 1);
        assert_eq!(sleepy_out.action, DecisionAction::ChangeExpression);
        assert_eq!(sleepy_out.target_expression.as_deref(), Some("sleepy"));

        let expr_out = local_fallback_decision(&base_ctx, false, false, 3);
        assert_eq!(expr_out.action, DecisionAction::ChangeExpression);
    }
}
