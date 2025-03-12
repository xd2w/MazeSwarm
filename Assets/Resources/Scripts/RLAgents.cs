using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class RLAgents : Agent
{
    [SerializeField]
    private float speed = 2000.0f;

    private Vector2 lastPos;
    private Rigidbody2D rigidbody;

    [Tooltip("whether this is a traning mode or gameplay mode")]
    public bool traningMode;

    private bool frozen = false;

    [Tooltip("The agents Camera")]
    public Camera agentCamera;

    private Vector3 origin = Vector3.zero;

    private GenerateMaze maze;

    private bool reset = false;

    public override void Initialize()
    {
        rigidbody = gameObject.GetComponent<Rigidbody2D>();
        maze = gameObject.GetComponentInParent<GenerateMaze>();
        // if not traning, no max steps, plays forever
        if (!traningMode) MaxStep = 0; // 0==infty

        Debug.Log("Initialised");

        lastPos = transform.position;

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        float x = Input.GetAxis("Horizontal");
        float y = Input.GetAxis("Vertical");

        // Debug.Log("Heuristic working");

        // float forward = 0;
        // float left = 0;

        // if (Input.GetKey(KeyCode.W)) forward = 1;
        // else if (Input.GetKey(KeyCode.S)) forward = -1;

        // if (Input.GetKey(KeyCode.A)) left = -1;
        // else if (Input.GetKey(KeyCode.D)) left = 1;

        ActionSegment<float> continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = x;
        continuousActionsOut[1] = y;

    }

    public override void OnActionReceived(ActionBuffers action)
    {
        // do nothing if frozen
        if (frozen) return;
        // Debug.Log("Action Received");

        ActionSegment<float> continuousActions = action.ContinuousActions;

        Vector2 move = new Vector2(continuousActions[0], continuousActions[1]);

        // rigidbody.AddForce(move * speed);
        rigidbody.linearVelocity = move * speed * Time.deltaTime;

        // apply rotation
        // transform.rotation = Quaternion;
    }

    // public override void CollectObservations(VectorSensor sensor)
    // {
    //     // base.CollectObservations(sensor);
    //     sensor.AddObservation(transform.position.x);
    //     sensor.AddObservation(transform.position.y);
    // }

    public override void OnEpisodeBegin()
    {
        // 0 out the velocity so it strats stationary
        rigidbody.linearVelocity = Vector2.zero;
        rigidbody.position = Vector2.zero;
        reset = true;
    }
    public void OnCollisionEnter2D(Collision2D collision)
    {
        AddReward(-0.01f);
    }

    public void OllisionStay2D(Collision2D collision)
    {
        AddReward(-0.01f);
    }

    private void Update()
    {
        // Draw a line from the breap tip to nearest flower for us
        Debug.DrawLine(transform.position, origin, Color.green);
        if (reset)
        {
            maze.ResetMaze();
            reset = false;
        }
    }

    private void FixedUpdate()
    {
        // if (Vector2.Distance(lastPos, transform.position) < 0.01)
        // {
        //     AddReward(-0.01f);
        // }
        AddReward(0.05f * Vector2.Distance(origin, transform.position) / (133 * Mathf.Sqrt(2)));

    }

    private void Reset()
    {
        rigidbody.linearVelocity = Vector2.zero;
        rigidbody.position = Vector2.zero;
        maze.ResetMaze();
    }

}
