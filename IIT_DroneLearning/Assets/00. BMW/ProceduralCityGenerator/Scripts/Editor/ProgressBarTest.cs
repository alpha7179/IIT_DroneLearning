using UnityEngine;
using UnityEditor;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// 진행률 표시 기능을 테스트하는 에디터 스크립트
    /// Task 13.4 검증용
    /// </summary>
    public class ProgressBarTest : EditorWindow
    {
        [MenuItem("City Generator/Test Progress Bar")]
        public static void ShowWindow()
        {
            GetWindow<ProgressBarTest>("Progress Bar Test");
        }

        private void OnGUI()
        {
            GUILayout.Label("진행률 표시 테스트", EditorStyles.boldLabel);
            GUILayout.Space(10);

            GUILayout.Label("이 테스트는 CityGenerator의 진행률 표시 기능을 검증합니다.");
            GUILayout.Label("Requirements: 11.4, 11.5");
            GUILayout.Space(10);

            if (GUILayout.Button("대규모 도시 생성 테스트 (진행률 표시)"))
            {
                TestProgressBarWithLargeCity();
            }

            GUILayout.Space(10);
            GUILayout.Label("테스트 방법:", EditorStyles.boldLabel);
            GUILayout.Label("1. 버튼을 클릭하여 대규모 도시 생성 시작");
            GUILayout.Label("2. 진행률 바가 표시되는지 확인");
            GUILayout.Label("3. 취소 버튼을 클릭하여 생성 중단 테스트");
            GUILayout.Label("4. 부분 생성된 도시가 정리되는지 확인");
        }

        private void TestProgressBarWithLargeCity()
        {
            // 씬에서 CityGenerator 찾기
            CityGenerator generator = FindAnyObjectByType<CityGenerator>();
            

            if (generator == null)
            {
                EditorUtility.DisplayDialog(
                    "오류",
                    "씬에 CityGenerator 컴포넌트를 찾을 수 없습니다.\n" +
                    "GameObject를 생성하고 CityGenerator 컴포넌트를 추가해주세요.",
                    "확인"
                );
                return;
            }

            // 대규모 도시 생성을 위한 파라미터 설정
            generator.minWidth = 30;
            generator.maxWidth = 40;
            generator.minDepth = 30;
            generator.maxDepth = 40;
            generator.buildingDensity = 0.8f;

            Debug.Log("ProgressBarTest: 대규모 도시 생성 시작 (30-40 x 30-40, 밀도 80%)");
            Debug.Log("ProgressBarTest: 진행률 바가 표시되고 취소 버튼이 작동하는지 확인하세요.");

            // 도시 생성 실행
            CityGenerationResult result = generator.GenerateCity();

            if (result.success)
            {
                EditorUtility.DisplayDialog(
                    "테스트 완료",
                    $"도시 생성 완료!\n\n" +
                    $"건물 수: {result.buildingCount}\n" +
                    $"생성 시간: {result.generationTime:F2}초\n\n" +
                    $"진행률 바가 정상적으로 표시되었나요?",
                    "확인"
                );
            }
            else
            {
                EditorUtility.DisplayDialog(
                    "테스트 결과",
                    $"도시 생성이 취소되었습니다.\n\n" +
                    $"오류 메시지: {result.errorMessage}\n" +
                    $"경과 시간: {result.generationTime:F2}초\n\n" +
                    $"부분 생성된 도시가 정리되었나요?",
                    "확인"
                );
            }
        }
    }
}
