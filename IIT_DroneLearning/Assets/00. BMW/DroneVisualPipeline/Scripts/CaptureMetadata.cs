using System;
using UnityEngine;

namespace DroneVisualPipeline
{
    /// <summary>
    /// 캡처 스텝별 메타데이터 직렬화 구조체.
    /// JsonUtility로 직렬화하여 step_NNNN_meta.json으로 저장된다.
    /// </summary>
    [Serializable]
    public class CaptureMetadata
    {
        public string timestamp;
        public int    episode;
        public int    step;
        public string droneRole;

        public SerializedVector3 dronePosition;
        public SerializedVector3 droneRotation;   // Euler angles
        public SerializedVector3 droneVelocity;
        public SerializedVector3 targetRelativePosition;

        public float[] sensorDistances;

        public float reward;
        public float cumulativeReward;
    }

    /// <summary>
    /// 에피소드 요약 직렬화 구조체.
    /// 에피소드 종료 시 episode_summary.json으로 저장된다.
    /// </summary>
    [Serializable]
    public class EpisodeSummary
    {
        public int    episode;
        public string droneRole;
        public int    totalSteps;
        public float  cumulativeReward;
        public string terminationReason;
        public int    capturedImageCount;
        public string startTime;
        public string endTime;
    }

    /// <summary>
    /// JsonUtility와 호환되는 Vector3 직렬화 헬퍼 구조체.
    /// (JsonUtility는 Vector3를 직렬화하지 않음)
    /// </summary>
    [Serializable]
    public struct SerializedVector3
    {
        public float x;
        public float y;
        public float z;

        public SerializedVector3(Vector3 v)
        {
            x = v.x;
            y = v.y;
            z = v.z;
        }

        public static implicit operator SerializedVector3(Vector3 v) => new SerializedVector3(v);
        public static implicit operator Vector3(SerializedVector3 s) => new Vector3(s.x, s.y, s.z);
    }
}
