import docker

class DockerSandbox:
    def __init__(self, image="python:3.10"):
        """Initialize the connection to Docker Desktop."""
        try:
            # Connects automatically to Docker Desktop on Windows
            self.client = docker.from_env() 
        except docker.errors.DockerException:
            raise Exception("Could not connect to Docker. Is Docker Desktop running?")
            
        self.image = image
        self.container = None
        
        print(f"Checking for Docker image '{self.image}'...")
        # Pull the image if you don't have it yet (might take a minute the very first time)
        self.client.images.pull(self.image)
        
    def start_container(self):
        """Spins up an isolated, background container."""
        print("Spinning up isolated sandbox container...")
        self.container = self.client.containers.run(
            self.image,
            command="tail -f /dev/null", # Keeps the container running in the background indefinitely
            detach=True,
            mem_limit="512m",  # Safety: Prevents the AI's code from crashing your PC's RAM
            network_disabled=True # Safety: Prevents the AI's code from accessing the internet
        )
        return self.container.id
        
    def execute_command(self, cmd: str) -> str:
        """Runs a bash/python command inside the active sandbox."""
        if not self.container:
            return "Error: Sandbox container is not running."
            
        exit_code, output = self.container.exec_run(cmd)
        
        # Return the string output so our AI agent can read it
        return output.decode("utf-8").strip()
        
    def cleanup(self):
        """Stops and deletes the container when finished."""
        if self.container:
            print("Cleaning up and destroying sandbox...")
            self.container.stop()
            self.container.remove()
            self.container = None

# ==========================================
# SANITY TEST
# ==========================================
if __name__ == "__main__":
    # 1. Initialize the Sandbox
    sandbox = DockerSandbox()
    
    try:
        # 2. Start the container
        sandbox.start_container()
        
        # 3. Ask the container to execute some Python code
        print("\n--- Testing Execution ---")
        test_script = 'python -c "print(\'Hello from inside the isolated Docker container!\')"'
        result = sandbox.execute_command(test_script)
        
        print(f"Output: {result}\n")
        
    finally:
        # 4. ALWAYS clean up, even if the test script crashes
        sandbox.cleanup()
        print("Sanity test complete.")