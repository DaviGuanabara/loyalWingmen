class Sensor:
    
    def __init__(self, parent_id: int, client_id: int):
        self.parent_id = parent_id
        self.client_id = client_id
        
        
    def read_data(self):
        raise NotImplementedError
    
    def update_data(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError