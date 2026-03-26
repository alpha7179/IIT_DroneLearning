// DroneDepthVisualize.shader
// URP 호환 Depth 시각화 셰이더
// _CameraDepthTexture → LinearEyeDepth → _MaxDistance 로 정규화 (0~1)
// Grayscale / Jet 컬러 램프 모드 지원

Shader "DroneVisualPipeline/DroneDepthVisualize"
{
    Properties
    {
        _MainTex        ("Source Texture",    2D)      = "white" {}
        _MaxDistance    ("Max Depth Distance", Float)  = 500.0
        _UseLinear      ("Use Linear Depth",  Int)     = 1
        _UseJetColorRamp("Use Jet Color Ramp",Int)     = 0
    }

    SubShader
    {
        Tags
        {
            "RenderType"  = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
        }

        Pass
        {
            Name "DroneDepthVisualize"
            ZTest Always
            ZWrite Off
            Cull Off

            HLSLPROGRAM

            #pragma vertex   vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float _MaxDistance;
                int   _UseLinear;
                int   _UseJetColorRamp;
            CBUFFER_END

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv         : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv         : TEXCOORD0;
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.uv         = IN.uv;
                return OUT;
            }

            // Jet 컬러 맵: 0(파랑) → 0.25(시안) → 0.5(초록) → 0.75(노랑) → 1(빨강)
            float3 JetColormap(float t)
            {
                float r = saturate(1.5 - abs(4.0 * t - 3.0));
                float g = saturate(1.5 - abs(4.0 * t - 2.0));
                float b = saturate(1.5 - abs(4.0 * t - 1.0));
                return float3(r, g, b);
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float2 uv = IN.uv;

                // URP _CameraDepthTexture 샘플링
                float rawDepth = SampleSceneDepth(uv);

                float normalized;
                if (_UseLinear)
                {
                    // LinearEyeDepth: 뷰 공간 선형 깊이 (카메라로부터의 거리, 미터)
                    float linearDepth = LinearEyeDepth(rawDepth, _ZBufferParams);
                    float maxDist     = max(_MaxDistance, 0.001);
                    normalized        = saturate(linearDepth / maxDist);
                }
                else
                {
                    // Raw: Z-buffer 비선형 값 그대로 사용 (0=near, 1=far in reversed-z)
                    normalized = rawDepth;
                }

                float3 color;
                if (_UseJetColorRamp)
                    color = JetColormap(normalized);
                else
                    color = float3(normalized, normalized, normalized);  // Grayscale

                return half4(color, 1.0);
            }

            ENDHLSL
        }
    }

    FallBack Off
}
