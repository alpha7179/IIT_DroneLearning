using UnityEngine;

namespace MyProject.DroneControl
{
    public class DroneController : MonoBehaviour
    {
        private Animator animator;

        // Start is called before the first frame update
        void Start()
        {
            // Get the Animator component attached to the drone
            animator = GetComponent<Animator>();

            if (animator == null)
            {
                Debug.LogError("Animator component not found on " + gameObject.name);
                return;
            }

            // Force the animation to start with "Idle" (or your neutral state) without any transition
            animator.Play("Idle", 0, 0f);  // Play Idle immediately at the start and ensure it loops until triggered
        }

        // Update is called once per frame
        void Update()
        {
            // Listen for key presses to trigger animations

            // Trigger take-off animation when the "T" key is pressed
            if (Input.GetKeyDown(KeyCode.T))
            {
                // Set the "TakeOffTrigger" trigger parameter in the Animator
                animator.SetTrigger("TakeOffTrigger");
            }

            // Trigger landing animation when the "L" key is pressed
            if (Input.GetKeyDown(KeyCode.L))
            {
                // Set the "LandTrigger" trigger parameter in the Animator
                animator.SetTrigger("LandTrigger");
            }
        }
    }
}
