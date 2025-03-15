using UnityEngine;
using System.Collections;
using System.Collections.Generic;


public class Room : MonoBehaviour
{
    public enum Directions
    {
        TOP, BOT, RIGHT, LEFT, NONE
    }

    [SerializeField]
    GameObject topWall;
    [SerializeField]
    GameObject botWall;
    [SerializeField]
    GameObject rightWall;
    [SerializeField]
    GameObject leftWall;

    [SerializeField]
    GameObject floorPanel;

    private RoomTrigger trigger;



    Dictionary<Directions, GameObject> walls = new Dictionary<Directions, GameObject>();

    public Vector2Int Index { get; set; }

    public bool visited { get; set; } = false;

    public bool covered { get; private set; } = false;

    Dictionary<Directions, bool> dirFlags = new Dictionary<Directions, bool>();

    private void Start()
    {
        walls[Directions.TOP] = topWall;
        walls[Directions.BOT] = botWall;
        walls[Directions.RIGHT] = rightWall;
        walls[Directions.LEFT] = leftWall;
        trigger = floorPanel.GetComponent<RoomTrigger>();

    }

    private void SetActivate(Directions dir, bool flag)
    {
        walls[dir].SetActive(flag);
    }

    public void SetDirFlag(Directions dir, bool flag)
    {
        dirFlags[dir] = flag;
        SetActivate(dir, flag);
    }

    public void resetCovered()
    {
        covered = false;
        trigger.ResetFloor();
    }

    public void flagCovered()
    {
        covered = true;
    }




}
