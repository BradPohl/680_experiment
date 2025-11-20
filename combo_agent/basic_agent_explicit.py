from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from typing import Literal
import os
import getpass
import base64
import json
import cv2
import mediapipe as mp
from dotenv import load_dotenv

load_dotenv()

# Verify that the GOOGLE_API_KEY is set
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Verify the API key is loaded
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key missing. Verify your .env file.")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.2,
    max_tokens=1024,
    timeout=600,
    max_retries=2,
)


@tool
def analyze_video(video_path: str) -> str:
    """
    Analyzes a Muay Thai video and returns the classified strikes based on video analysis.
    This tool reads the video file and sends it to the LLM for analysis.
    
    Args:
        video_path: Path to the video file to analyze
        
    Returns:
        A string containing the predicted strikes in JSON array format
    """
    # Verify video file exists
    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"
    
    try:
        # Read and encode the video file
        mime_type = "video/quicktime" if video_path.endswith(".mov") else "video/mp4"
        
        with open(video_path, 'rb') as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")
        
        # System prompt for Muay Thai strike classification
        question = """You are an expert Muay Thai strike classifier. Your task is to analyze videos of Muay Thai strikes and identify ALL strikes performed in chronological order.

CRITICAL: Watch the ENTIRE video from start to finish. Multiple strikes may occur in sequence. You must identify EVERY strike, not just one.

Available strike types with detailed descriptions:
- jab: A quick, straight forward punch with the lead (front) hand, arm extends straight forward from shoulder
- cross: A powerful straight punch with the rear hand, arm extends straight forward from shoulder, body rotates
- lead_hook: A curved punch with the lead hand, arm swings in an arc from the side
- rear_hook: A curved punch with the rear hand, arm swings in an arc from the side
- lead_uppercut: An upward punch with the lead hand, arm moves upward from below
- rear_uppercut: An upward punch with the rear hand, arm moves upward from below
- roundhouse: A circular kick where the leg swings in an arc, typically hitting with the shin or instep. The leg comes from the side and wraps around. Also called "low kick" when targeting the leg.
- switch_kick: A roundhouse kick performed after switching stance (stepping forward with rear leg first)
- teep: A front push kick where the foot pushes straight forward, like a front kick. The leg extends forward, not sideways. This is DIFFERENT from a roundhouse kick.
- knee: A knee strike where the knee is driven upward into a target
- switch_knee: A knee strike after switching stance
- lead_elbow: An elbow strike with the lead arm, elbow strikes forward or sideways
- rear_elbow: An elbow strike with the rear arm, elbow strikes forward or sideways

IMPORTANT DISTINCTIONS:
- teep vs roundhouse: A teep is a straight forward push kick. A roundhouse is a circular/sideways kick. They are VERY different movements.
- Watch for multiple strikes in sequence - a combo might be jab, then cross, then roundhouse kick.

Analyze the video frame by frame if needed. Identify ALL strikes in the exact order they appear. Return ONLY a JSON array of strike names, like: ["jab", "cross", "roundhouse"]. Do not include any other text."""
        
        # Create a message with video using LangChain format
        messages = [HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "media", "mime_type": mime_type, "data": video_data}
        ])]
        
        # Use the same LLM instance that the agent uses
        response = llm.invoke(messages)
        
        result = response.content
        
        # Debug: Print what we got back
        print(f"[DEBUG] analyze_video returned: {result[:200]}...")
        
        return result
    except Exception as e:
        return f"Error analyzing video: {str(e)}"


@tool
def extract_pose_data(video_path: str) -> str:
    """
    Extracts raw pose keypoint coordinates from a Muay Thai video using MediaPipe Pose.
    Returns frame-by-frame pose landmarks with coordinates (x, y, z, visibility).
    
    Args:
        video_path: Path to the video file to analyze
        
    Returns:
        A JSON string containing pose data with frame numbers, landmark names, and coordinates
    """
    # Verify video file exists
    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"
    
    try:
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"Error: Could not open video file {video_path}"
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # MediaPipe landmark names (33 landmarks)
        landmark_names = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
            "right_ear", "mouth_left", "mouth_right", "left_shoulder",
            "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
            "right_wrist", "left_pinky", "right_pinky", "left_index",
            "right_index", "left_thumb", "right_thumb", "left_hip",
            "right_hip", "left_knee", "right_knee", "left_ankle",
            "right_ankle", "left_heel", "right_heel", "left_foot_index",
            "right_foot_index"
        ]
        
        # Store pose data for all frames
        pose_data = {
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "video_path": video_path
            },
            "frames": []
        }
        
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB (MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = pose.process(rgb_frame)
            
            # Extract landmarks for this frame
            frame_data = {
                "frame_number": frame_number,
                "landmarks": {}
            }
            
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_name = landmark_names[idx] if idx < len(landmark_names) else f"landmark_{idx}"
                    frame_data["landmarks"][landmark_name] = {
                        "x": float(landmark.x),
                        "y": float(landmark.y),
                        "z": float(landmark.z),
                        "visibility": float(landmark.visibility)
                    }
            else:
                # No pose detected in this frame
                frame_data["landmarks"] = {}
            
            pose_data["frames"].append(frame_data)
            frame_number += 1
        
        # Release resources
        cap.release()
        pose.close()
        
        # Return as JSON string
        return json.dumps(pose_data, indent=2)
        
    except Exception as e:
        return f"Error extracting pose data: {str(e)}"


# Define the State
class AgentState(MessagesState):
    """State for the Muay Thai strike classification agent."""
    pass


# Define the agent node using create_agent
from langchain.agents import create_agent

strike_agent = create_agent(
    model=llm,
    tools=[analyze_video, extract_pose_data],
    system_prompt="""You are a Muay Thai video analysis agent that classifies strikes in videos. You have access to two tools:

1. analyze_video(video_path): Analyzes the video directly using multimodal AI vision capabilities. This provides a comprehensive view of the entire video including visual context, movement patterns, and strike execution.

2. extract_pose_data(video_path): Extracts raw pose keypoint coordinates (x, y, z, visibility) for all 33 body landmarks frame-by-frame using MediaPipe. This provides precise body position data that you can analyze to identify strike patterns based on joint positions and movements.

Available strike types:
- jab: A quick, straight punch with the lead hand
- cross: A straight punch with the rear hand
- lead_hook: A hook punch with the lead hand
- rear_hook: A hook punch with the rear hand
- lead_uppercut: An uppercut punch with the lead hand
- rear_uppercut: An uppercut punch with the rear hand
- roundhouse: A roundhouse kick (typically with the rear leg)
- switch_kick: A roundhouse kick after switching stance
- teep: A front push kick
- knee: A knee strike
- switch_knee: A knee strike after switching stance
- lead_elbow: An elbow strike with the lead arm
- rear_elbow: An elbow strike with the rear arm

You can choose which tool to use based on the task:
- Use analyze_video for quick, comprehensive analysis using visual understanding
- Use extract_pose_data when you need precise body position data to analyze joint angles, limb positions, and movement trajectories

You may use both tools if needed - for example, extract pose data first for detailed analysis, then use video analysis for confirmation or additional context.

CRITICAL INSTRUCTIONS:
- Always use at least one tool to analyze videos - do not try to answer without using a tool
- When using analyze_video, the tool will return a JSON array - use that directly
- If the tool returns only one strike but you suspect there are more, use extract_pose_data to get detailed frame-by-frame analysis
- Return your final strike classifications as a JSON array in chronological order, like: ["jab", "cross", "roundhouse"]
- Make sure to identify ALL strikes in the video, not just one
- Pay special attention to distinguishing between teep (straight forward push) and roundhouse (circular/sideways kick)"""
)


# Define the agent node function
def agent_node(state: AgentState) -> Command[Literal["__end__"]]:
    """
    Agent node that processes the state and invokes the strike classification agent.
    
    Returns:
        Command: Command to update state and route to END
    """
    # Invoke the agent with the current state
    result = strike_agent.invoke(state)
    
    # Return command to update state and end
    return Command(
        update={
            "messages": result["messages"]
        },
        goto=END
    )


# Build the explicit graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

# Compile the graph
graph = builder.compile()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default to first combo video for testing
        video_path = "data/combos/1.mov"
    
    print(f"Agent analyzing video: {video_path}")
    print("=" * 50)
    
    # Invoke the graph with the video path
    result = graph.invoke(
        {"messages": [HumanMessage(content=f"Classify the strikes in this video: {video_path}")]}
    )
    
    print("\nAgent Response:")
    print("=" * 50)
    print(result["messages"][-1].content)

