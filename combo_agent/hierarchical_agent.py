from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from typing_extensions import TypedDict
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
def analyze_video(video_path: str, classification_level: str = "fine") -> str:
    """
    Analyzes a Muay Thai video and returns the classified strikes based on video analysis.
    This tool reads the video file and sends it to the LLM for analysis.
    
    Args:
        video_path: Path to the video file to analyze
        classification_level: "coarse" for category-level (punch/kick/knee/elbow) or "fine" for specific strikes
        
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
        
        if classification_level == "coarse":
            # System prompt for COARSE classification only
            question = """You are a Muay Thai strike category classifier. Your task is to identify the COARSE category of each strike in a video.

CRITICAL INSTRUCTIONS:
1. Watch the ENTIRE video from start to finish
2. Multiple strikes may occur in sequence - you MUST identify EVERY strike
3. Count each strike separately - if there are 3 strikes, return 3 categories
4. Return categories in the exact order they appear

Coarse categories (ONLY return these 4 options):
- punch: Any strike using the hands/fists (arm/hand movements, fist contact, punching motion)
- kick: Any strike using the feet/legs (leg/foot movements, leg extension, kicking motion)
- knee: Any strike using the knee (knee driving upward into target)
- elbow: Any strike using the elbow (elbow striking forward or sideways)

Visual indicators:
- punch: Arm extends forward/sideways, fist makes contact, hand/arm movement
- kick: Leg extends, foot/leg makes contact, leg swings or pushes
- knee: Knee drives upward, leg bends and extends upward
- elbow: Elbow strikes forward or sideways, arm bends at elbow

IMPORTANT: 
- A combo might be: punch, then punch, then kick = ["punch", "punch", "kick"]
- Do NOT combine multiple strikes into one
- Do NOT return specific strike types like "jab" or "roundhouse"
- ONLY return "punch", "kick", "knee", or "elbow"

Analyze the video frame by frame if needed. Identify the coarse category for ALL strikes in chronological order.
Return ONLY a JSON array of coarse categories, like: ["punch", "punch", "kick"]"""
        else:
            # System prompt for FINE classification
            question = """You are an expert Muay Thai strike classifier. Your task is to analyze videos of Muay Thai strikes and identify all strikes performed in chronological order.

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

Analyze the video carefully and identify all strikes in the order they appear. Return your response as a JSON array of strike names, like: ["jab", "cross", "roundhouse"]"""
        
        # Create a message with video using LangChain format
        messages = [HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "media", "mime_type": mime_type, "data": video_data}
        ])]
        
        # Use the same LLM instance that the agent uses
        response = llm.invoke(messages)
        
        result = response.content
        
        # Debug: Print what we got back
        print(f"[DEBUG] analyze_video ({classification_level}) returned: {result[:200]}...")
        
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


# Define the State with coarse classification result
class HierarchicalState(MessagesState):
    """State for the hierarchical strike classification agent."""
    coarse_categories: list = []  # List of coarse categories (punch, kick, knee, elbow)
    fine_strikes: list = []  # List of fine-grained strike classifications
    video_path: str = ""


# Create the coarse classifier agent
from langchain.agents import create_agent

coarse_agent = create_agent(
    model=llm,
    tools=[analyze_video, extract_pose_data],
    system_prompt="""You are a Muay Thai strike category classifier. Your task is to identify the COARSE category of each strike in a video.

Coarse categories:
- punch: Any strike using the hands/fists (jab, cross, hook, uppercut)
- kick: Any strike using the feet/legs (roundhouse, teep, switch_kick)
- knee: Any strike using the knee
- elbow: Any strike using the elbow

You have access to two tools:
1. analyze_video(video_path, classification_level="coarse"): Analyzes the video and returns COARSE categories only. Always use classification_level="coarse" when calling this tool.
2. extract_pose_data(video_path): Extracts raw pose keypoint coordinates

IMPORTANT: When using analyze_video, always pass classification_level="coarse" to get category-level classification, not specific strike types.

Watch the ENTIRE video and identify the coarse category for EACH strike in chronological order. 
Look for:
- punch: Arm/hand movements, fist contact
- kick: Leg/foot movements, leg extension
- knee: Knee driving upward
- elbow: Elbow striking forward or sideways

Return your response as a JSON array of coarse categories in order, like: ["punch", "punch", "kick"]
Only return the coarse categories, not specific strike types."""
)


# Create fine classifier agents for each category
punch_fine_agent = create_agent(
    model=llm,
    tools=[analyze_video, extract_pose_data],
    system_prompt="""You are a Muay Thai punch classifier. You identify specific punch types.

Available punch types:
- jab: A quick, straight forward punch with the lead (front) hand
- cross: A powerful straight punch with the rear hand, body rotates
- lead_hook: A curved punch with the lead hand, arm swings in an arc from the side
- rear_hook: A curved punch with the rear hand, arm swings in an arc from the side
- lead_uppercut: An upward punch with the lead hand, arm moves upward from below
- rear_uppercut: An upward punch with the rear hand, arm moves upward from below

You have access to:
1. analyze_video(video_path): Analyzes the video directly
2. extract_pose_data(video_path): Extracts pose keypoints

Given that a PUNCH was identified, determine the specific punch type. 
Return your response as a JSON array of specific punch types, like: ["jab", "cross"]"""
)


kick_fine_agent = create_agent(
    model=llm,
    tools=[analyze_video, extract_pose_data],
    system_prompt="""You are a Muay Thai kick classifier. You identify specific kick types.

Available kick types:
- roundhouse: A circular kick where the leg swings in an arc, typically hitting with the shin or instep. The leg comes from the side and wraps around.
- switch_kick: A roundhouse kick performed after switching stance (stepping forward with rear leg first)
- teep: A front push kick where the foot pushes straight forward, like a front kick. The leg extends forward, not sideways.

You have access to:
1. analyze_video(video_path): Analyzes the video directly
2. extract_pose_data(video_path): Extracts pose keypoints

Given that a KICK was identified, determine the specific kick type.
IMPORTANT: A teep is a straight forward push kick. A roundhouse is a circular/sideways kick. They are VERY different.
Return your response as a JSON array of specific kick types, like: ["roundhouse", "teep"]"""
)


knee_fine_agent = create_agent(
    model=llm,
    tools=[analyze_video, extract_pose_data],
    system_prompt="""You are a Muay Thai knee strike classifier. You identify specific knee strike types.

Available knee types:
- knee: A knee strike where the knee is driven upward into a target
- switch_knee: A knee strike after switching stance

You have access to:
1. analyze_video(video_path): Analyzes the video directly
2. extract_pose_data(video_path): Extracts pose keypoints

Given that a KNEE strike was identified, determine if it's a regular knee or switch_knee.
Return your response as a JSON array of specific knee types, like: ["knee", "switch_knee"]"""
)


elbow_fine_agent = create_agent(
    model=llm,
    tools=[analyze_video, extract_pose_data],
    system_prompt="""You are a Muay Thai elbow strike classifier. You identify specific elbow strike types.

Available elbow types:
- lead_elbow: An elbow strike with the lead arm, elbow strikes forward or sideways
- rear_elbow: An elbow strike with the rear arm, elbow strikes forward or sideways

You have access to:
1. analyze_video(video_path): Analyzes the video directly
2. extract_pose_data(video_path): Extracts pose keypoints

Given that an ELBOW strike was identified, determine if it's lead_elbow or rear_elbow.
Return your response as a JSON array of specific elbow types, like: ["lead_elbow", "rear_elbow"]"""
)


# Define node functions
def coarse_node(state: HierarchicalState) -> Command[Literal["fine"]]:
    """
    Coarse classification node - identifies strike categories.
    """
    print("\n[COARSE] Classifying strike categories...")
    
    # Invoke the coarse agent
    result = coarse_agent.invoke(state)
    
    # Extract coarse categories from response
    # Handle both string and list content formats
    agent_response_raw = result["messages"][-1].content
    print(f"[COARSE] Response (raw): {agent_response_raw}")
    
    # Extract text from content (could be string or list of content blocks)
    if isinstance(agent_response_raw, list):
        # Extract text from content blocks
        agent_response = ""
        for block in agent_response_raw:
            if isinstance(block, dict) and 'text' in block:
                agent_response += block['text']
            elif isinstance(block, str):
                agent_response += block
        print(f"[COARSE] Extracted text: {agent_response}")
    else:
        agent_response = str(agent_response_raw)
    
    # Try to parse JSON array from response
    import re
    json_match = re.search(r'\[.*?\]', agent_response)
    coarse_categories = []
    
    if json_match:
        try:
            coarse_categories = json.loads(json_match.group())
            print(f"[COARSE] Parsed categories: {coarse_categories}")
        except json.JSONDecodeError:
            print("[COARSE] Could not parse JSON, using fallback")
            # Fallback: try to extract categories from text
            if "punch" in agent_response.lower():
                coarse_categories.append("punch")
            if "kick" in agent_response.lower():
                coarse_categories.append("kick")
            if "knee" in agent_response.lower():
                coarse_categories.append("knee")
            if "elbow" in agent_response.lower():
                coarse_categories.append("elbow")
    else:
        print("[COARSE] No JSON array found, using fallback")
        # Fallback: try to extract categories from text
        if "punch" in agent_response.lower():
            coarse_categories.append("punch")
        if "kick" in agent_response.lower():
            coarse_categories.append("kick")
        if "knee" in agent_response.lower():
            coarse_categories.append("knee")
        if "elbow" in agent_response.lower():
            coarse_categories.append("elbow")
    
    return Command(
        update={
            "messages": result["messages"],
            "coarse_categories": coarse_categories,
            "video_path": state.get("video_path", "")
        },
        goto="fine"
    )


def fine_node(state: HierarchicalState) -> Command[Literal["__end__"]]:
    """
    Fine classification node - identifies specific strike types based on coarse categories.
    """
    print("\n[FINE] Classifying specific strike types...")
    
    coarse_categories = state.get("coarse_categories", [])
    video_path = state.get("video_path", "")
    
    if not coarse_categories:
        return Command(
            update={
                "messages": [AIMessage(content="No coarse categories identified. Cannot perform fine classification.")]
            },
            goto="__end__"
        )
    
    fine_strikes = []
    
    # Group consecutive strikes of the same category
    current_category = None
    category_group = []
    category_groups = []
    
    for category in coarse_categories:
        if category == current_category:
            category_group.append(category)
        else:
            if category_group:
                category_groups.append((current_category, len(category_group)))
            current_category = category
            category_group = [category]
    
    if category_group:
        category_groups.append((current_category, len(category_group)))
    
    print(f"[FINE] Processing {len(category_groups)} category groups: {category_groups}")
    
    # Classify each group
    for category, count in category_groups:
        print(f"\n[FINE] Classifying {count} {category}(es)...")
        
        # Select appropriate fine agent
        if category == "punch":
            fine_agent = punch_fine_agent
        elif category == "kick":
            fine_agent = kick_fine_agent
        elif category == "knee":
            fine_agent = knee_fine_agent
        elif category == "elbow":
            fine_agent = elbow_fine_agent
        else:
            print(f"[FINE] Unknown category: {category}, skipping")
            continue
        
        # Create message for fine classification
        fine_message = HumanMessage(
            content=f"Classify the {count} {category}(es) in this video: {video_path}. Return exactly {count} specific {category} type(s) as a JSON array."
        )
        
        # Invoke fine agent
        fine_result = fine_agent.invoke({"messages": [fine_message]})
        fine_response_raw = fine_result["messages"][-1].content
        print(f"[FINE] {category} response (raw): {fine_response_raw}")
        
        # Extract text from content (could be string or list of content blocks)
        if isinstance(fine_response_raw, list):
            # Extract text from content blocks
            fine_response = ""
            for block in fine_response_raw:
                if isinstance(block, dict) and 'text' in block:
                    fine_response += block['text']
                elif isinstance(block, str):
                    fine_response += block
            print(f"[FINE] {category} extracted text: {fine_response}")
        else:
            fine_response = str(fine_response_raw)
        
        # Parse fine classification
        import re
        json_match = re.search(r'\[.*?\]', fine_response)
        if json_match:
            try:
                category_strikes = json.loads(json_match.group())
                # Ensure we have the right number
                if len(category_strikes) >= count:
                    fine_strikes.extend(category_strikes[:count])
                else:
                    # Repeat the last one if we need more
                    while len(fine_strikes) < len(coarse_categories):
                        fine_strikes.extend(category_strikes)
                    fine_strikes = fine_strikes[:len(coarse_categories)]
            except json.JSONDecodeError:
                print(f"[FINE] Could not parse JSON for {category}")
                # Fallback: use category name
                fine_strikes.extend([category] * count)
        else:
            # Fallback: use category name
            fine_strikes.extend([category] * count)
    
    # Ensure we have the same number as coarse categories
    if len(fine_strikes) != len(coarse_categories):
        print(f"[FINE] Warning: Mismatch in counts. Coarse: {len(coarse_categories)}, Fine: {len(fine_strikes)}")
        # Pad or truncate as needed
        while len(fine_strikes) < len(coarse_categories):
            fine_strikes.append(coarse_categories[len(fine_strikes)])
        fine_strikes = fine_strikes[:len(coarse_categories)]
    
    # Format final response with clear JSON array
    final_classification_json = json.dumps(fine_strikes)
    final_response = f"Coarse categories: {coarse_categories}\nFine strikes: {fine_strikes}\n\nFinal classification: {final_classification_json}"
    
    print(f"[FINE] Final strikes: {fine_strikes}")
    print(f"[FINE] Final JSON: {final_classification_json}")
    
    return Command(
        update={
            "messages": [AIMessage(content=final_response)],
            "fine_strikes": fine_strikes  # Also store in state for easier access
        },
        goto="__end__"
    )


# Build the hierarchical graph
builder = StateGraph(HierarchicalState)
builder.add_node("coarse", coarse_node)
builder.add_node("fine", fine_node)
builder.add_edge(START, "coarse")
builder.add_edge("coarse", "fine")
builder.add_edge("fine", END)

# Compile the graph
hierarchical_graph = builder.compile()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default to first combo video for testing
        video_path = "data/combos/1.mov"
    
    print(f"Hierarchical agent analyzing video: {video_path}")
    print("=" * 60)
    
    # Invoke the hierarchical graph
    result = hierarchical_graph.invoke(
        {
            "messages": [HumanMessage(content=f"Classify the strikes in this video: {video_path}")],
            "video_path": video_path
        }
    )
    
    print("\n" + "=" * 60)
    print("Final Result:")
    print("=" * 60)
    print(result["messages"][-1].content)

