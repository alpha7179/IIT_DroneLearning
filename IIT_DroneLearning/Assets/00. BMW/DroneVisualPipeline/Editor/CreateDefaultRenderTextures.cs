using UnityEngine;
using UnityEditor;

namespace DroneVisualPipeline.Editor
{
    /// <summary>
    /// DroneVisionSystem / DroneDepthSystem 용 기본 RenderTexture 에셋을 생성하는 에디터 유틸리티.
    /// 메뉴: Tools > DroneVisualPipeline > Create Default RenderTextures
    /// </summary>
    public static class CreateDefaultRenderTextures
    {
        private const string BasePath = "Assets/00. BMW/DroneVisualPipeline";

        [MenuItem("Tools/DroneVisualPipeline/Create Default RenderTextures")]
        public static void Create()
        {
            CreateRT("DroneVision_RT_84x84", 84, 84, 24, RenderTextureFormat.ARGB32);
            CreateRT("DroneDepth_RT_84x84",  84, 84, 24, RenderTextureFormat.ARGB32);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log("[DroneVisualPipeline] 기본 RenderTexture 에셋 생성 완료.");
        }

        private static void CreateRT(string name, int w, int h, int depth, RenderTextureFormat format)
        {
            string path = $"{BasePath}/{name}.renderTexture";
            var existing = AssetDatabase.LoadAssetAtPath<RenderTexture>(path);
            if (existing != null)
            {
                Debug.Log($"[DroneVisualPipeline] 이미 존재: {path}");
                return;
            }

            var rt = new RenderTexture(w, h, depth, format);
            rt.name = name;
            AssetDatabase.CreateAsset(rt, path);
            Debug.Log($"[DroneVisualPipeline] 생성됨: {path}");
        }
    }
}
