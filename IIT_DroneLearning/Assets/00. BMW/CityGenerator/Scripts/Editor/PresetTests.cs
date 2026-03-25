using UnityEngine;
using UnityEditor;

namespace CityGenerator.Editor
{
    /// <summary>
    /// CityGenerator의 프리셋 저장 및 로드 기능을 테스트하는 에디터 스크립트
    /// </summary>
    public static class PresetTests
    {
        [MenuItem("City Generator/Test Save Preset")]
        public static void TestSavePreset()
        {
            Debug.Log("=== Starting Save Preset Tests ===");

            // 테스트용 GameObject 생성
            GameObject testObj = new GameObject("TestCityGenerator");
            CityGenerator generator = testObj.AddComponent<CityGenerator>();

            // Test 1: 기본 파라미터로 프리셋 저장
            Debug.Log("\n--- Test 1: Save Preset with Default Parameters ---");
            generator.unitDistance = 1.5f;
            generator.minWidth = 15;
            generator.maxWidth = 25;
            generator.minDepth = 12;
            generator.maxDepth = 22;
            generator.buildingWidth = 1.2f;
            generator.buildingDepth = 1.3f;
            generator.minBuildingHeight = 8.0f;
            generator.maxBuildingHeight = 25.0f;
            generator.buildingSpacing = 1.5f;
            generator.buildingDensity = 0.8f;
            generator.randomSeed = 54321;
            generator.minimapResolution = MinimapResolution.Resolution1024;

            string presetName1 = "TestPreset_Default";
            generator.SavePreset(presetName1);
            
            // 프리셋이 생성되었는지 확인
            string assetPath1 = $"Assets/CityPresets/{presetName1}.asset";
            CityParameters loadedPreset1 = AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath1);
            
            if (loadedPreset1 != null)
            {
                Debug.Log($"✓ 프리셋 저장 성공: {assetPath1}");
                
                // 파라미터 값 검증
                bool parametersMatch = 
                    Mathf.Approximately(loadedPreset1.unitDistance, generator.unitDistance) &&
                    loadedPreset1.minWidth == generator.minWidth &&
                    loadedPreset1.maxWidth == generator.maxWidth &&
                    loadedPreset1.minDepth == generator.minDepth &&
                    loadedPreset1.maxDepth == generator.maxDepth &&
                    Mathf.Approximately(loadedPreset1.buildingWidth, generator.buildingWidth) &&
                    Mathf.Approximately(loadedPreset1.buildingDepth, generator.buildingDepth) &&
                    Mathf.Approximately(loadedPreset1.minBuildingHeight, generator.minBuildingHeight) &&
                    Mathf.Approximately(loadedPreset1.maxBuildingHeight, generator.maxBuildingHeight) &&
                    Mathf.Approximately(loadedPreset1.buildingSpacing, generator.buildingSpacing) &&
                    Mathf.Approximately(loadedPreset1.buildingDensity, generator.buildingDensity) &&
                    loadedPreset1.randomSeed == generator.randomSeed &&
                    loadedPreset1.minimapResolution == generator.minimapResolution;
                
                if (parametersMatch)
                {
                    Debug.Log("✓ 모든 파라미터가 정확히 저장됨");
                }
                else
                {
                    Debug.LogError("✗ 일부 파라미터가 올바르게 저장되지 않음");
                    Debug.Log($"  unitDistance: {loadedPreset1.unitDistance} vs {generator.unitDistance}");
                    Debug.Log($"  minWidth: {loadedPreset1.minWidth} vs {generator.minWidth}");
                    Debug.Log($"  randomSeed: {loadedPreset1.randomSeed} vs {generator.randomSeed}");
                }
            }
            else
            {
                Debug.LogError($"✗ 프리셋 저장 실패: {assetPath1}");
            }

            // Test 2: 다른 파라미터로 프리셋 저장
            Debug.Log("\n--- Test 2: Save Preset with Different Parameters ---");
            generator.unitDistance = 2.0f;
            generator.minWidth = 5;
            generator.maxWidth = 10;
            generator.buildingDensity = 0.5f;
            generator.randomSeed = 99999;
            generator.minimapResolution = MinimapResolution.Resolution256;

            string presetName2 = "TestPreset_Custom";
            generator.SavePreset(presetName2);
            
            string assetPath2 = $"Assets/CityPresets/{presetName2}.asset";
            CityParameters loadedPreset2 = AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath2);
            
            if (loadedPreset2 != null)
            {
                Debug.Log($"✓ 프리셋 저장 성공: {assetPath2}");
                Debug.Log($"  unitDistance: {loadedPreset2.unitDistance}");
                Debug.Log($"  buildingDensity: {loadedPreset2.buildingDensity}");
                Debug.Log($"  randomSeed: {loadedPreset2.randomSeed}");
                Debug.Log($"  minimapResolution: {loadedPreset2.minimapResolution}");
            }
            else
            {
                Debug.LogError($"✗ 프리셋 저장 실패: {assetPath2}");
            }

            // Test 3: 빈 이름으로 저장 시도 (실패해야 함)
            Debug.Log("\n--- Test 3: Save Preset with Empty Name (Should Fail) ---");
            generator.SavePreset("");
            Debug.Log("빈 이름으로 저장 시도 완료 (오류 메시지가 출력되어야 함)");

            // Test 4: 기존 프리셋 덮어쓰기
            Debug.Log("\n--- Test 4: Overwrite Existing Preset ---");
            generator.unitDistance = 3.0f;
            generator.buildingDensity = 0.9f;
            generator.SavePreset(presetName1); // 첫 번째 프리셋 덮어쓰기
            
            CityParameters overwrittenPreset = AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath1);
            if (overwrittenPreset != null)
            {
                Debug.Log($"✓ 프리셋 덮어쓰기 성공");
                Debug.Log($"  새로운 unitDistance: {overwrittenPreset.unitDistance} (예상: 3.0)");
                Debug.Log($"  새로운 buildingDensity: {overwrittenPreset.buildingDensity} (예상: 0.9)");
                
                bool overwriteSuccess = 
                    Mathf.Approximately(overwrittenPreset.unitDistance, 3.0f) &&
                    Mathf.Approximately(overwrittenPreset.buildingDensity, 0.9f);
                
                if (overwriteSuccess)
                {
                    Debug.Log("✓ 덮어쓰기가 정확히 수행됨");
                }
                else
                {
                    Debug.LogError("✗ 덮어쓰기가 올바르게 수행되지 않음");
                }
            }

            // 정리
            Object.DestroyImmediate(testObj);
            
            // 테스트 프리셋 파일 삭제 (선택사항)
            Debug.Log("\n--- Cleanup: Deleting Test Presets ---");
            if (AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath1) != null)
            {
                AssetDatabase.DeleteAsset(assetPath1);
                Debug.Log($"삭제됨: {assetPath1}");
            }
            if (AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath2) != null)
            {
                AssetDatabase.DeleteAsset(assetPath2);
                Debug.Log($"삭제됨: {assetPath2}");
            }
            AssetDatabase.Refresh();

            Debug.Log("\n=== Save Preset Tests Complete ===");
        }

        [MenuItem("City Generator/Test Load Preset")]
        public static void TestLoadPreset()
        {
            Debug.Log("=== Starting Load Preset Tests ===");

            // 테스트용 GameObject 생성
            GameObject testObj = new GameObject("TestCityGenerator");
            CityGenerator generator = testObj.AddComponent<CityGenerator>();

            // Test 1: 프리셋 저장 후 로드
            Debug.Log("\n--- Test 1: Save and Load Preset ---");
            
            // 원본 파라미터 설정
            generator.unitDistance = 2.5f;
            generator.minWidth = 20;
            generator.maxWidth = 30;
            generator.buildingDensity = 0.75f;
            generator.randomSeed = 12345;
            
            Debug.Log("원본 파라미터:");
            Debug.Log($"  unitDistance: {generator.unitDistance}");
            Debug.Log($"  minWidth: {generator.minWidth}");
            Debug.Log($"  maxWidth: {generator.maxWidth}");
            Debug.Log($"  buildingDensity: {generator.buildingDensity}");
            Debug.Log($"  randomSeed: {generator.randomSeed}");
            
            // 프리셋 저장
            string presetName = "TestPreset_LoadTest";
            generator.SavePreset(presetName);
            
            // 파라미터 변경
            generator.unitDistance = 1.0f;
            generator.minWidth = 5;
            generator.maxWidth = 10;
            generator.buildingDensity = 0.3f;
            generator.randomSeed = 99999;
            
            Debug.Log("\n변경된 파라미터:");
            Debug.Log($"  unitDistance: {generator.unitDistance}");
            Debug.Log($"  minWidth: {generator.minWidth}");
            Debug.Log($"  maxWidth: {generator.maxWidth}");
            Debug.Log($"  buildingDensity: {generator.buildingDensity}");
            Debug.Log($"  randomSeed: {generator.randomSeed}");
            
            // 프리셋 로드
            string assetPath = $"Assets/CityPresets/{presetName}.asset";
            CityParameters preset = AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath);
            
            if (preset != null)
            {
                generator.LoadPreset(preset);
                
                Debug.Log("\n로드된 파라미터:");
                Debug.Log($"  unitDistance: {generator.unitDistance}");
                Debug.Log($"  minWidth: {generator.minWidth}");
                Debug.Log($"  maxWidth: {generator.maxWidth}");
                Debug.Log($"  buildingDensity: {generator.buildingDensity}");
                Debug.Log($"  randomSeed: {generator.randomSeed}");
                
                // 원본 값과 비교
                bool parametersRestored = 
                    Mathf.Approximately(generator.unitDistance, 2.5f) &&
                    generator.minWidth == 20 &&
                    generator.maxWidth == 30 &&
                    Mathf.Approximately(generator.buildingDensity, 0.75f) &&
                    generator.randomSeed == 12345;
                
                if (parametersRestored)
                {
                    Debug.Log("✓ 모든 파라미터가 정확히 복원됨");
                }
                else
                {
                    Debug.LogError("✗ 일부 파라미터가 올바르게 복원되지 않음");
                }
            }
            else
            {
                Debug.LogError($"✗ 프리셋을 찾을 수 없음: {assetPath}");
            }

            // Test 2: null 프리셋 로드 시도 (오류 처리 확인)
            Debug.Log("\n--- Test 2: Load Null Preset (Should Fail Gracefully) ---");
            generator.LoadPreset(null);
            Debug.Log("null 프리셋 로드 시도 완료 (오류 메시지가 출력되어야 함)");

            // 정리
            Object.DestroyImmediate(testObj);
            
            // 테스트 프리셋 파일 삭제
            if (AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath) != null)
            {
                AssetDatabase.DeleteAsset(assetPath);
                Debug.Log($"삭제됨: {assetPath}");
            }
            AssetDatabase.Refresh();

            Debug.Log("\n=== Load Preset Tests Complete ===");
        }
    }
}
