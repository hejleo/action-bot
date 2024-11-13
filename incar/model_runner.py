from transformers import AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch
import sentencepiece
import json
import os

class ActionArranger:
    """
    A class that handles natural language processing of user prompts and matches them to appropriate nodes.
    Uses T5 transformer model for text processing and sentence transformers for similarity matching.
    """
    
    def __init__(self):
        """
        Initialize the ActionArranger with necessary ML models and configurations.
        Sets up T5 model for text processing and SentenceTransformer for similarity matching.
        Handles device selection (MPS, CUDA, or CPU) and loads node definitions.
        """
        try:
            # Using FLAN-T5-small model for lighter resource usage
            self.model_name = "google/flan-t5-xxl" # Changed to small google/flan-t5-small for faster inference and prototyping
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="/root/.cache/huggingface"
            )
            
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir="/root/.cache/huggingface",
                torch_dtype=torch.float16 if self.device.type != "cpu" else None,
                device_map="auto",
                use_cache=True,
                low_cpu_mem_usage=True
            )
            
            # Using a smaller sentence transformer model
            self.similarity_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L12-v2",
                cache_folder="/root/.cache/huggingface",
                device=self.device.type
            )
            
            # Set model to evaluation mode
            self.model.eval()
            self.similarity_model.eval()
            
            # Load node definitions from JSON file
            self.node_definitions = self.load_node_definitions()
            
            self.node_categories = {
                "Event Nodes": [
                    "OnVariableChange", "OnKeyPress", "OnKeyRelease", "OnClick",
                    "OnWindowResize", "OnMouseEnter", "OnMouseLeave", "OnTimer"
                ],
                "Action Nodes": [
                    "Console", "Alert", "Log", "Assign", "SendRequest",
                    "Navigate", "Save", "Delete", "PlaySound", "PauseSound", "StopSound"
                ],
                "Data Nodes": [
                    "FetchData", "StoreData", "UpdateData", "DeleteData", "CacheData"
                ]
            }
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    def load_node_definitions(self):
        """
        Load node definitions from a JSON file containing node descriptions and metadata.
        
        Returns:
            dict: Dictionary containing node definitions, or empty dict if loading fails
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, 'node_descriptions.json')
            
            if not os.path.exists(json_path):
                print(f"JSON file not found at: {json_path}")
                return {}
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading node definitions: {str(e)}")
            return {}

    def rearrange_actions(self, prompt: str, max_length: int = None) -> str:
        """
        Process a natural language prompt to extract and rearrange actions in logical order.
        
        Args:
            prompt (str): The input text to process
            max_length (int, optional): Maximum length of generated output
            
        Returns:
            str: Comma-separated string of rearranged actions
        """
        if max_length is None:
            max_length = len(prompt) + 90

        instruction = (
                "Read the given sentence carefully and follow these detailed steps:\n"
                "1. Identify and extract all key action words, especially verbs and their associated phrases. Do not neglect any important action words.\n"
                "2. Pay attention to the context to determine the logical and temporal order of these actions.\n"
                "3. Rephrase each action into a clear and concise phrase, starting with a capital letter. Include necessary details like time delays or conditions.\n"
                "4. List the actions in the precise order they occur, ensuring they reflect both logical progression and temporal sequence.\n"
                "5. Return the actions as a single string with each action separated by a comma and a space.\n"
                f"Input Sentence: {prompt}\n"
        )
        
        print("\n=== Rearranging Actions ===")
        print(f"Input prompt: {prompt}")
        
        try:
            inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=4,
                do_sample=False,
                temperature=1.0,
                early_stopping=True,
                use_cache=True
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output
            result = result.strip()
            # Split by periods or commas and clean each part
            parts = [part.strip() for part in result.replace('.', ',').split(',')]
            # Filter out empty parts and join with commas
            result = ", ".join([part for part in parts if part])
            
            print(f"Processed result: {result}")
            return result
            
        except Exception as e:
            print(f"Error in rearrange_actions: {str(e)}")
            raise

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text strings using sentence transformers.
        
        Args:
            text1 (str): First text for comparison
            text2 (str): Second text for comparison
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Ensure inputs are strings and handle empty/None values
        text1 = str(text1) if text1 else ""
        text2 = str(text2) if text2 else ""
        
        # Skip empty comparisons
        if not text1 or not text2:
            print(f"Skipping empty comparison: '{text1}' or '{text2}' is empty")
            return 0.0
        
            # Convert entire strings to lowercase for better comparison
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # print(f"\n=== Similarity Calculation Debug ===")
        # print(f"Input text1: '{text1}' (length: {len(text1)})")
        # print(f"Input text2: '{text2}' (length: {len(text2)})")
        
        # Encode full texts for comparison
        embedding1 = self.similarity_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.similarity_model.encode(text2, convert_to_tensor=True)
        
        # print(f"Embedding1 shape: {embedding1.shape}")
        # print(f"Embedding2 shape: {embedding2.shape}")
        
        similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0), 
            embedding2.unsqueeze(0)
        )
        
        print(f"Final similarity between '{text1}' and '{text2}': {similarity.item():.4f}")
        print("=== End Similarity Calculation ===\n")
        return similarity.item()


    def find_affine_nodes(self, actions):
        """
        Find nodes that are semantically similar to given actions and generate their descriptions.
        
        Args:
            actions (list): List of action phrases to match
            
        Returns:
            tuple: (list of matching node names, list of node descriptions with HTML formatting)
        """
        affine_nodes = []
        node_descriptions = []
        
        # Create copy of categories to modify
        node_categories = {k: v[:] for k, v in self.node_categories.items()}
        
        # Find affine nodes for each action
        for action in actions:
            max_similarity = -1
            best_node = None
            
            for category, nodes in node_categories.items():
                for node in nodes:
                    
                    similarity = self.calculate_similarity(action, node)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_node = node
                        
            if max_similarity > 0.4:
                affine_nodes.append(best_node)
                # Get node description from JSON definitions
                if best_node in self.node_definitions:
                    node_info = self.node_definitions[best_node]
                    
                    html_description = f"""
                    <div class="node-description">
                        <h3>{best_node}</h3>
                        <p><strong>Type:</strong> {node_info.get('type', 'N/A')}</p>
                        <p><strong>Category:</strong> {node_info.get('category', 'N/A')}</p>
                        <p><strong>Description:</strong> {node_info.get('description', 'No description available')}</p>
                    </div>
                    """
                    
                    # Only add scope and triggers if they exist
                    if 'scope' in node_info:
                        html_description = html_description[:-6]  # Remove closing div
                        html_description += f'<p><strong>Scope:</strong> {", ".join(node_info["scope"])}</p></div>'
                        
                    if 'triggers' in node_info:
                        html_description = html_description[:-6]  # Remove closing div
                        html_description += f'<p><strong>Triggers:</strong> {", ".join(node_info["triggers"])}</p></div>'
                    
                    node_descriptions.append({
                        "node": best_node,
                        "html": html_description,
                        "raw_data": node_info
                    })
                # Remove used node from available nodes
                for nodes in node_categories.values():
                    if best_node in nodes:
                        nodes.remove(best_node)
        
        return affine_nodes, node_descriptions

    def __call__(self, prompt: str) -> list:
        """
        Make the class callable. Process a prompt and return matching nodes.
        
        Args:
            prompt (str): Input text to process
            
        Returns:
            list: List of matching nodes
        """
        return []