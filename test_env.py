from env import EmailTriageEnv
from models import Action  # important

# Create environment
env = EmailTriageEnv(task_id="task_easy")

# Reset environment
obs = env.reset()
print("\nInitial Observation:\n", obs)

done = False

while not done:
    # Create a dummy action (adjust fields based on your Action model)
    action = Action(
        category="urgent",
        priority="high",
        route_to="inbox",     
        draft_reply=None      
    )

    obs, reward, done, info = env.step(action)

    print("\nStep Result:")
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    print("Next Obs:", obs)

print("\nFinal State:")
print(env.state())