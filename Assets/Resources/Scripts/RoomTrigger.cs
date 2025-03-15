using UnityEngine;

public class RoomTrigger : MonoBehaviour
{
    SpriteRenderer floor;
    Room room;

    void Start()
    {
        floor = gameObject.GetComponent<SpriteRenderer>();
        room = gameObject.GetComponentInParent<Room>();

    }

    void OnTriggerExit2D(Collider2D collision)
    {
        floor.color = Color.red;
        room.flagCovered();

    }

    public void ResetFloor()
    {
        floor.color = new Color(0.2745098f, 0.2745098f, 0.2745098f);
    }
}
