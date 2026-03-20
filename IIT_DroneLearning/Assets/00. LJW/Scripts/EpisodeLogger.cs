using UnityEngine;
using System.IO;
using System.Text;

/// <summary>
/// EpisodeLogger — eval 전용 에피소드 결과 기록기
///
/// EvaderAgent와 같은 GameObject에 붙인다.
/// Inference Only 모드에서 N 에피소드 실행 후 CSV 저장 + Play 자동 중지.
///
/// 사용법:
///   1. Behavior Type = Inference Only, Model = evader_s0_flat_v3.onnx
///   2. 이 컴포넌트를 EvaderAgent 오브젝트에 추가
///   3. Inspector에서 Max Episodes, Output Path 설정
///   4. Play → 완료 시 자동 중지 + CSV 저장
/// </summary>
public class EpisodeLogger : MonoBehaviour
{
    [Header("Eval Settings")]
    [Tooltip("평가할 에피소드 수")]
    [SerializeField] private int _maxEpisodes = 50;

    [Tooltip("CSV 저장 경로 (비우면 프로젝트 루트/eval_result.csv)")]
    [SerializeField] private string _outputPath = "";

    [Header("Debug")]
    [SerializeField] private bool _logEachEpisode = true;

    // 종료 유형 (EvaderAgent에서 호출)
    public enum TermType { Goal, Timeout, Crash, Captured }

    private int _episodeCount = 0;
    private int _goalCount    = 0;
    private int _timeoutCount = 0;
    private int _crashCount   = 0;
    private int _capturedCount = 0;

    private StringBuilder _csv;

    private void Awake()
    {
        _csv = new StringBuilder();
        _csv.AppendLine("episode_id,termination,duration_steps");
    }

    /// <summary>EvaderAgent의 CheckTerminationConditions에서 호출</summary>
    public void LogEpisode(TermType termination, int durationSteps)
    {
        _episodeCount++;

        switch (termination)
        {
            case TermType.Goal:     _goalCount++;     break;
            case TermType.Timeout:  _timeoutCount++;  break;
            case TermType.Crash:    _crashCount++;    break;
            case TermType.Captured: _capturedCount++; break;
        }

        _csv.AppendLine($"{_episodeCount},{termination.ToString().ToLower()},{durationSteps}");

        if (_logEachEpisode)
            Debug.Log($"[Eval] #{_episodeCount}/{_maxEpisodes} | {termination} | steps={durationSteps} | goal={_goalCount}");

        if (_episodeCount >= _maxEpisodes)
            Finish();
    }

    private void Finish()
    {
        float goalRate    = _goalCount    / (float)_episodeCount * 100f;
        float timeoutRate = _timeoutCount / (float)_episodeCount * 100f;
        float crashRate   = _crashCount   / (float)_episodeCount * 100f;
        float captureRate = _capturedCount / (float)_episodeCount * 100f;

        Debug.Log($"[Eval] ===== RESULT ({_episodeCount} episodes) =====");
        Debug.Log($"[Eval] goal_reach_rate  : {goalRate:F1}%  (target >=50%)");
        Debug.Log($"[Eval] timeout_rate     : {timeoutRate:F1}%");
        Debug.Log($"[Eval] crash_rate       : {crashRate:F1}%");
        Debug.Log($"[Eval] capture_rate     : {captureRate:F1}%");
        Debug.Log($"[Eval] =======================================");

        SaveCsv();

#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#endif
    }

    private void SaveCsv()
    {
        string path = string.IsNullOrEmpty(_outputPath)
            ? Path.Combine(Application.dataPath, "../eval_result.csv")
            : _outputPath;

        path = Path.GetFullPath(path);
        File.WriteAllText(path, _csv.ToString(), Encoding.UTF8);
        Debug.Log($"[Eval] CSV saved → {path}");
    }
}
