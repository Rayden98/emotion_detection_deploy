from locust import SequentialTaskSet, HttpUser, task

class DetectorTask(SequentialTaskSet):
    @task
    def detection(self):
        with open("test_image.jpg", "rb") as image:
            self.client.post(
                "/detect",
                files={"im": image}
            )
            
            
class LoadTester(HttpUser):
    host="https://emotions-detection-b0ad110258b3.herokuapp.com"
    tasks=[DetectorTask]