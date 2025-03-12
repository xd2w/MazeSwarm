using UnityEngine;

public class Agnet : MonoBehaviour
{


    [SerializeField]
    private float speed = 2.0f;

    // [SerializeField]
    private Rigidbody2D rigidBody;
    // Start is called once before the first execution of Update after the MonoBehaviour is created

    void Start()
    {
        rigidBody = gameObject.GetComponent<Rigidbody2D>();

    }

    void FixedUpdate()
    {
        float x = Input.GetAxis("Horizontal");
        float y = Input.GetAxis("Vertical");

        Vector3 movement = new Vector3(x, y, 0);
        rigidBody.linearVelocity = movement * speed * Time.deltaTime;

    }

    void OnCollisionEnter(Collision collision)
    {
        if (gameObject.CompareTag("wall"))
        {
            rigidBody.linearVelocity = Vector3.zero;
        }
    }

}
