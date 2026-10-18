import os
import re
from typing import Dict, Any, List, Optional, Annotated

from pydantic import BaseModel
from datetime import datetime
import requests

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_community.utilities import GoogleSerperAPIWrapper

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.errors import NodeInterrupt

from IPython.display import display, Image

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("TONGYI_API_KEY")
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["LINKEDIN_COOKIE"] = os.getenv("LINKEDIN_COOKIE")

llm = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

search = GoogleSerperAPIWrapper()

# 职位要求模型
class JobRequirements(BaseModel):
    # 关于职位要求的具体细节
    title: str
    company_description: str
    job_requirements: List[str]
    candidate_responsibilities: List[str]
    candidate_qualifications: List[str]
    company_benefits: List[str]
    interview_date: datetime
    preferred_country: str
    years_experience: int
    linkedin_profiles: Optional[List[str]] = None
    skills_required: List[str]
    salary_range: str

# 候选人档案模型
class CandidateProfile(BaseModel):
    # 关于这个候选人的详细信息
    name: str
    linkedin_url: str
    title: str
    location: str
    cv_content: Optional[str] = None
    cv_score: Optional[float] = None
    status: str = "new"  # 'new', 'contacted', 'cv_received', 'approved', 'rejected'
    feedback: Optional[str] = None
    source: str = "direct"  # 'direct' or 'search'
    skill_matches: Optional[List[str]] = None
    skill_gaps: Optional[List[str]] = None
    message_sent: Optional[str] = None


# 主状态模型
class RecruitmentState(BaseModel):
    # 招聘流程的整体状况
    phase: str
    messages: Annotated[List[BaseMessage], add_messages]
    job_requirements: Optional[JobRequirements]
    job_description: Optional[str]
    job_description_approved: bool
    candidates: List[CandidateProfile]
    linkedin_process_complete: bool
    cv_analysis_complete: bool
    interview_questions: Optional[List[str]]
    interview_questions_approved: bool

# 使用Google Serper API在LinkedIn上搜索候选人
def search_linkedin_candidates(
        job_title: str,
        location: str,
        skills: List[str],
        limit: int = 5
) -> List[Dict[str, str]]:
    """使用Google Serper API在LinkedIn上搜索候选人"""
    search = GoogleSerperAPIWrapper()
    search_query = f"""site:linkedin.com/in/
            {job_title}
            {location}
            {' '.join(skills)}"""

    try:
        results = search.results(search_query)
        candidates = []
        for result in results.get('organic', [])[:limit]:
            if "linkedin.com/in/" in result.get("link", ""):
                candidates.append({
                    "linkedin_url": result["link"],
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", "")
                })
        return candidates
    except Exception as e:
        return [{"error": str(e)}]

# 获取详细的LinkedIn个人资料信息
def get_linkedin_profile(profile_url: str) -> Dict[str, Any]:
    """获取详细的LinkedIn个人资料信息"""
    headers = {
        'cookie': os.getenv("LINKEDIN_COOKIE"),
        'accept': 'application/json'
    }
    try:
        profile_id = profile_url.split('/in/')[-1].split('/')[0]
        api_url = f"https://www.linkedin.com/voyager/api/identity/profiles/{profile_id}/profileView"
        response = requests.get(api_url, headers=headers)
        data = response.json()

        return {
            "name": f"{data.get('firstName', '')} {data.get('lastName', '')}",
            "title": data.get('headline', ''),
            "location": data.get('locationName', ''),
            "skills": [skill.get('name', '') for skill in data.get('skills', [])]
        }
    except Exception as e:
        return {"error": str(e)}

# 向那位在LinkedIn上找到的候选人发送消息，介绍这个工作机会。
def send_linkedin_message(profile_url: str, job_details: str) -> Dict[str, str]:
    """向那位在LinkedIn上找到的候选人发送消息，介绍这个工作机会。"""
    headers = {
        'cookie': os.getenv("LINKEDIN_COOKIE"),
        'accept': 'application/json'
    }
    try:
        profile_id = profile_url.split('/in/')[-1].split('/')[0]

        # 生成个性化的消息
        message_prompt = [
            SystemMessage(
                content="Generate a personalized LinkedIn outreach message for a job opportunity. Ask them to submit their CV directly through LinkedIn messages if interested."),
            HumanMessage(content=f"Job Details:\n{job_details}")
        ]
        message_content = llm.invoke(message_prompt).content

        # 通过LinkedIn API发送消息
        message_endpoint = f"https://www.linkedin.com/voyager/api/messaging/conversations"
        payload = {
            "recipients": [profile_id],
            "message": message_content
        }
        response = requests.post(message_endpoint, headers=headers, json=payload)

        return {
            "status": "sent",
            "profile_id": profile_id,
            "message": message_content
        }
    except Exception as e:
        return {"error": str(e)}

"""定义工作流节点：包括需求收集、职位描述、简历分析等节点"""
def requirements_gathering(state: RecruitmentState) -> Dict[str, Any]:
    """
    初始化或继续需求收集过程。

    参数:
        state: 当前的招聘工作流程状态

    返回值:
        包含状态更新的字典
    """
    if not state.messages:  # 使用点表示法而不是字典表示法
        # 使用JobRequirements模型来输入职位要求信息。
        job_requirements = JobRequirements(
            title=input("What is the job title? "),
            company_description=input("Provide a description of the company: "),
            job_requirements=input("List the job requirements (comma-separated): ").split(","),
            candidate_responsibilities=input("List the candidate responsibilities (comma-separated): ").split(","),
            candidate_qualifications=input("List the candidate qualifications (comma-separated): ").split(","),
            company_benefits=input("List the company benefits (comma-separated): ").split(","),
            interview_date=datetime.strptime(
                input("What is the interview date (YYYY-MM-DD)? "),
                "%Y-%m-%d"
            ),
            preferred_country=input("What is the preferred country for the role? "),
            years_experience=int(input("What is the required years of experience? ")),
            linkedin_profiles=[
                p.strip()
                for p in input("Provide any LinkedIn profile URLs (comma-separated, optional): ").split(",")
                if p.strip()
            ],
            skills_required=input("List the required skills (comma-separated): ").split(","),
            salary_range=input("What is the salary range for the role? ")
        )

        return {
            "messages": [
                SystemMessage(content="You are an HR assistant gathering detailed job requirements."),
                HumanMessage(content="Let's begin gathering the job requirements. What is the job title?")
            ],
            "job_requirements": job_requirements,
            "phase": "requirements_gathering"
        }
    else:
        return {"phase": state.phase}  # Return current state if messages exist

def generate_job_desc(state: RecruitmentState) -> Dict[str, Any]:
    """
    生成包含人工审核的职位描述。

    参数：
        state：当前招聘工作流程的状态

    返回：
        一个包含状态更新的字典

    异常：
        NodeInterrupt：需要人工审核生成的描述
    """
    if not state.job_requirements:
        raise ValueError("Job requirements missing")

    messages = [
        SystemMessage(content="""Create a professional and compelling job description with:
            1. About the Company
            2. Role Overview
            3. Key Responsibilities
            4. Required Qualifications
            5. What We Offer (Benefits)
            6. Location and Work Mode"""),
        HumanMessage(content=str(state.job_requirements))
    ]

    response = llm.invoke(messages)

    raise NodeInterrupt(
        f"Please review the generated job description:\n\n{response.content}"
    )

def linkedin_process(state: RecruitmentState) -> Dict[str, Any]:
    """处理在LinkedIn上搜索或联系候选人的流程"""
    candidates = []

    job_details = f"""
    Role: {state.job_requirements.title}
    Company: {state.job_requirements.company_description}
    Location: {state.job_requirements.preferred_country}
    Description: {state.job_description}
    """

    if state.job_requirements.linkedin_profiles:
        # 处理已提供的LinkedIn资料
        for profile_url in state.job_requirements.linkedin_profiles:
            profile = get_linkedin_profile(profile_url)

            if "error" not in profile:
                message_result = send_linkedin_message(profile_url=profile_url, job_details=job_details)

                if "error" not in message_result:
                    candidates.append(
                        CandidateProfile(
                            **profile,
                            linkedin_url=profile_url,
                            status="contacted",
                            message_sent=message_result["message"]
                        )
                    )

    else:
        # 使用Serper API来搜索候选人
        search_results = search_linkedin_candidates(
            job_title=state.job_requirements.title,
            location=state.job_requirements.preferred_country,
            skills=state.job_requirements.skills_required,
        )

        for result in search_results:
            if "error" not in result:
                profile = get_linkedin_profile(result["linkedin_url"])

                if "error" not in profile:
                    message_result = send_linkedin_message(
                        profile_url=result["linkedin_url"],
                        job_details=job_details
                    )

                    if "error" not in message_result:
                        candidates.append(
                            CandidateProfile(
                                **profile,
                                linkedin_url=result["linkedin_url"],
                                status="contacted",
                                source="search",
                                message_sent=message_result["message"]
                            )
                        )

    return {
        "candidates": candidates,
        "linkedin_process_complete": True,
        "phase": "analyze_cv"
    }

def analyze_cv(state: RecruitmentState) -> Dict[str, Any]:
    # 执行任务，分析候选人的简历，并发送相关信息
    if not state.candidates:
        return {"phase": "complete", "cv_analysis_complete": True}

    candidate = state.candidates[-1]

    messages = [
        SystemMessage(content="""Analyze the CV against job requirements. Score from 0-10 on:
        1. Skills Match
        2. Experience Level
        3. Overall Fit

        Provide detailed feedback and clear recommendation.
        End your response with exactly one line in this format: Overall Score: X.X/10"""),
        HumanMessage(content=f"""
        Job Requirements:
        {state.job_requirements.model_dump_json()}

        CV Content:
        {candidate.cv_content or 'Not provided'}
        """)
    ]

    analysis = llm.invoke(messages)
    match = re.search(r"Overall Score:\s*(\d+\.?\d*)", analysis.content)
    score = float(match.group(1)) if match else 0.0

    if score >= 7.0:
        candidate.status = "approved"
        message = f"Congratulations! You've been selected for an interview on {state.job_requirements.interview_date}"
    else:
        candidate.status = "rejected"
        message = "Thank you for your application. Unfortunately..."

    candidate.cv_score = score
    candidate.feedback = analysis.content

    # 通过LinkedIn发送消息
    send_linkedin_message(profile_url=candidate.linkedin_url, job_details=message)

    return {
        "candidates": list(state.candidates),
        "cv_analysis_complete": True,
        "phase": "prepare_interview" if candidate.status == "approved" else "complete"
    }

# 面试准备环节
def prepare_interview(state: RecruitmentState) -> Dict[str, Any]:
    """生成带有人工审核的面试问题"""
    messages = [
        SystemMessage(content="""Generate 10 interview questions covering:
            - Technical Skills (4)
            - Experience (3)
            - Problem Solving (3)"""),
        HumanMessage(content=f"""
            Position: {state.job_requirements.title}
            Required Skills: {', '.join(state.job_requirements.skills_required)}
            """)
    ]

    response = llm.invoke(messages)

    # 将打断工作流，让人工审查介入
    raise NodeInterrupt(
        f"Please review the interview questions:\n\n{response.content}"
    )

# 创建招聘工作流
def create_recruitment_workflow():
    """创建招聘的Workflow Graph"""
    workflow = StateGraph(RecruitmentState)
    memory = MemorySaver()

    workflow.add_node("requirements_gathering", requirements_gathering)
    workflow.add_node("generate_job_desc", generate_job_desc)
    workflow.add_node("linkedin_process", linkedin_process)
    workflow.add_node("analyze_cv", analyze_cv)
    workflow.add_node("prepare_interview", prepare_interview)

    workflow.add_edge(START, "requirements_gathering")
    workflow.add_edge("requirements_gathering", "generate_job_desc")

    def route_after_job_desc(state: RecruitmentState):
        return "linkedin_process" if state.job_description_approved else "generate_job_desc"

    def route_after_cv(state: RecruitmentState):
        if not state.candidates:
            return END
        return "prepare_interview" if state.candidates[-1].status == "approved" else END

    workflow.add_conditional_edges(
        "generate_job_desc",
        route_after_job_desc,
        ["linkedin_process", "generate_job_desc"]
    )

    workflow.add_edge("linkedin_process", "analyze_cv")
    workflow.add_conditional_edges(
        "analyze_cv",
        route_after_cv,
        ["prepare_interview", END]
    )
    workflow.add_edge("prepare_interview", END)

    graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=["generate_job_desc", "prepare_interview"],
    )

    # 生成并展示Mermaid可视化效果
    print("""graph TD
            Start --> requirements_gathering
            requirements_gathering --> generate_job_desc
            generate_job_desc -->|Approved| linkedin_process
            generate_job_desc -->|Not Approved| generate_job_desc
            linkedin_process --> analyze_cv
            analyze_cv -->|CV-Good| prepare_interview
            analyze_cv -->|CV-Bad| End
            prepare_interview -->|Approved| End
            prepare_interview -->|Not Approved| prepare_interview
        """)
    display(Image(graph.get_graph().draw_mermaid_png()))

    return graph

def get_job_requirements():
    job_title = input("What is the job title? ")
    company_description = input("Provide a description of the company: ")
    job_requirements = input("List the job requirements (comma-separated): ").split(",")
    candidate_responsibilities = input("List the candidate responsibilities (comma-separated): ").split(",")
    candidate_qualifications = input("List the candidate qualifications (comma-separated): ").split(",")
    company_benefits = input("List the company benefits (comma-separated): ").split(",")
    interview_date = input("What is the interview date (YYYY-MM-DD)? ")
    preferred_country = input("What is the preferred country for the role? ")
    years_experience = int(input("What is the required years of experience? "))
    linkedin_profiles = input("Provide any LinkedIn profile URLs (comma-separated, optional): ").split(",")
    skills_required = input("List the required skills (comma-separated): ").split(",")
    salary_range = input("What is the salary range for the role? ")

    return JobRequirements(
        title=job_title,
        company_description=company_description,
        job_requirements=job_requirements,
        candidate_responsibilities=candidate_responsibilities,
        candidate_qualifications=candidate_qualifications,
        company_benefits=company_benefits,
        interview_date=interview_date,
        preferred_country=preferred_country,
        years_experience=years_experience,
        linkedin_profiles=linkedin_profiles,
        skills_required=skills_required,
        salary_range=salary_range,
    )

# 创建招聘工作流程实例
recruitment_workflow = create_recruitment_workflow()

# 设置测试配置
config = {"configurable": {"thread_id": "test_1"}}

# 定义招聘流程的初始化状态
initial_state = {
    "phase": "requirements_gathering",
    "messages": [],
    "job_requirements": None,
    "job_description": None,
    "job_description_approved": False,
    "candidates": [],
    "linkedin_process_complete": False,
    "cv_analysis_complete": False,
    "interview_questions": None,
    "interview_questions_approved": False
}

# 运行工作流并处理中断
try:
    for event in recruitment_workflow.stream(initial_state, config, stream_mode="values"):
        print("\nNew State Update:")
        print(f"Phase: {event.get('phase')}")
        if event.get('messages'):
            print(f"Latest Message: {event['messages'][-1].content}")
except Exception as e:
    print(f"\nHuman Review Required: {str(e)}")

    # 处理职位描述审批的例子
    recruitment_workflow.update_state(
        config,
        {"job_description_approved": True}
    )

    # 继续执行工作流程
    for event in recruitment_workflow.stream(None, config, stream_mode="values"):
        print("\nNew State Update:")
        print(f"Phase: {event.get('phase')}")
        if event.get('messages'):
            print(f"Latest Message: {event['messages'][-1].content}")

