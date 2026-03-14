using System;
using System.Collections;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

public class DronePhysicsPlayModeTests
{
    private GameObject _droneObject;
    private Component _dronePhysics;
    private Type _dronePhysicsType;
    private Rigidbody _rigidbody;

    [UnitySetUp]
    public IEnumerator SetUp()
    {
        _dronePhysicsType = ResolveDronePhysicsType();
        Assert.IsNotNull(_dronePhysicsType, "DronePhysics type should be available at runtime.");

        _droneObject = new GameObject("DronePhysicsTest");
        _droneObject.AddComponent<BoxCollider>();

        _rigidbody = _droneObject.AddComponent<Rigidbody>();
        _rigidbody.useGravity = true;

        _dronePhysics = _droneObject.AddComponent(_dronePhysicsType);
        _droneObject.transform.position = new Vector3(0f, 3f, 0f);
        _droneObject.transform.rotation = Quaternion.identity;

        yield return null;
        InvokeDronePhysicsMethod("ResetPhysics");
        yield return new WaitForFixedUpdate();
    }

    [UnityTearDown]
    public IEnumerator TearDown()
    {
        if (_droneObject != null)
            UnityEngine.Object.Destroy(_droneObject);

        yield return null;
    }

    [UnityTest]
    public IEnumerator ResetPhysics_AfterPoseChange_DoesNotIntroduceYawSpin()
    {
        _droneObject.transform.SetPositionAndRotation(
            new Vector3(0f, 4f, 0f),
            Quaternion.Euler(0f, 75f, 0f)
        );

        InvokeDronePhysicsMethod("ResetPhysics");
        float initialYaw = _droneObject.transform.eulerAngles.y;

        for (int i = 0; i < 10; i++)
            yield return new WaitForFixedUpdate();

        float yawDelta = Mathf.Abs(Mathf.DeltaAngle(initialYaw, _droneObject.transform.eulerAngles.y));
        Assert.Less(yawDelta, 2f, "Reset should sync yaw target to the spawned pose.");
    }

    [UnityTest]
    public IEnumerator YawCommand_ChangesHeading()
    {
        float initialYaw = _droneObject.transform.eulerAngles.y;

        for (int i = 0; i < 25; i++)
        {
            InvokeDronePhysicsMethod("SetCommand", 0f, 0f, 0f, 0.6f);
            yield return new WaitForFixedUpdate();
        }

        float yawDelta = Mathf.Abs(Mathf.DeltaAngle(initialYaw, _droneObject.transform.eulerAngles.y));
        Assert.Greater(yawDelta, 5f, "Yaw input should produce a visible heading change.");
    }

    [UnityTest]
    public IEnumerator ZeroInputHorizontalDamping_ReducesHorizontalSpeed()
    {
        _rigidbody.linearVelocity = new Vector3(3f, 0f, 0f);
        float initialSpeed = new Vector2(_rigidbody.linearVelocity.x, _rigidbody.linearVelocity.z).magnitude;

        for (int i = 0; i < 20; i++)
        {
            InvokeDronePhysicsMethod("SetCommand", 0f, 0f, 0f, 0f);
            yield return new WaitForFixedUpdate();
        }

        float finalSpeed = new Vector2(_rigidbody.linearVelocity.x, _rigidbody.linearVelocity.z).magnitude;
        Assert.Less(finalSpeed, initialSpeed, "Horizontal speed should decay after input release.");
    }

    private static Type ResolveDronePhysicsType()
    {
        foreach (Assembly assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            Type type = assembly.GetType("DronePhysics");
            if (type != null)
                return type;
        }

        return null;
    }

    private void InvokeDronePhysicsMethod(string methodName, params object[] args)
    {
        var method = _dronePhysicsType.GetMethod(methodName);
        Assert.IsNotNull(method, $"DronePhysics should contain method '{methodName}'.");
        method.Invoke(_dronePhysics, args);
    }
}
