"""
Optimized RAG-Based Component Inspector for Arcelik-Beko Washing Machines
Combines similarity search (RAG) with component type detection for multi-step workflow
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import logging

from embeddings.clip_embedder import CLIPEmbedder

logger = logging.getLogger(__name__)


class RAGComponentInspector:
    """
    RAG-based inspector for Arcelik-Beko washing machines that:
    1. Detects component type (cabinet panel/detergent dispenser/front wall/general surface)
    2. Uses similarity search to detect damage (existing RAG approach)
    3. Validates correct component for each inspection step
    """
    
    # Component type definitions - Arcelik-Beko Washing Machine Parts
    COMPONENT_TYPES = {
        'cabinet_panel': 'Cabinet_Panels_Damaged',
        'detergent_dispenser': 'Detergent_Dispenser_Damaged', 
        'front_wall': 'Front_Wall_Damaged',
        'general_surface': 'Scratches_General'
    }
    
    # Text descriptions for CLIP-based discrimination
    COMPONENT_TEXT_DESCRIPTIONS = {
        'cabinet_panel': "washing machine side panel, metal cabinet wall with dents and scratches, outer casing surface damage",
        'detergent_dispenser': "detergent dispenser with rusty screws, soap drawer assembly with corrosion, dispenser housing with mechanical damage",
        'front_wall': "washing machine front door panel, front wall assembly with damage, door frame with dents or scratches",
        'general_surface': "general surface scratches on appliance, cosmetic paint damage, minor wear marks on washing machine exterior"
    }
    
    # Step to component mapping (can be customized for multi-step workflow)
    STEP_COMPONENTS = {
        1: 'cabinet_panel',
        2: 'detergent_dispenser',
        3: 'front_wall',
        4: 'general_surface'
    }
    
    def __init__(self, embedder: CLIPEmbedder):
        """
        Initialize the RAG inspector
        
        Args:
            embedder: CLIP embedder instance
        """
        self.embedder = embedder
        self.component_databases = {}
        self.good_casting_database = None  # For hybrid casting damage detection
        self.is_ready = False
        
    def build_component_databases(self, parts_images_dir: Path) -> None:
        """
        Build separate RAG databases for each washing machine component type
        This allows component-specific similarity search
        
        Args:
            parts_images_dir: Directory with component images
        """
        logger.info("🔨 Building RAG databases for Arcelik-Beko washing machine components...")
        
        self.component_databases = {
            'cabinet_panel': {'embeddings': [], 'paths': [], 'metadata': []},
            'detergent_dispenser': {'embeddings': [], 'paths': [], 'metadata': []},
            'front_wall': {'embeddings': [], 'paths': [], 'metadata': []},
            'general_surface': {'embeddings': [], 'paths': [], 'metadata': []}
        }
        
        # No need for good/ok references for now - all images are damaged
        # This can be added later if OK reference images are provided
        
        # Process each component directory
        for comp_dir in parts_images_dir.iterdir():
            if not comp_dir.is_dir():
                continue
            
            # Map directory to component type
            if "Cabinet_Panels" in comp_dir.name:
                comp_type = 'cabinet_panel'
            elif "Detergent_Dispenser" in comp_dir.name:
                comp_type = 'detergent_dispenser'
            elif "Front_Wall" in comp_dir.name:
                comp_type = 'front_wall'
            elif "Scratches_General" in comp_dir.name:
                comp_type = 'general_surface'
            else:
                logger.warning(f"Unknown component directory: {comp_dir.name}")
                continue
            
            # Collect images
            image_paths = list(comp_dir.glob("*.jpg")) + list(comp_dir.glob("*.jpeg")) + list(comp_dir.glob("*.png")) + list(comp_dir.glob("*.Jpeg"))
            
            if not image_paths:
                logger.warning(f"No images found in {comp_dir.name}")
                continue
            
            logger.info(f"  Processing {len(image_paths)} {comp_type} images from {comp_dir.name}...")
            
            # Generate embeddings
            embeddings = self.embedder.encode_batch_images(image_paths)
            
            # Store in component-specific database
            self.component_databases[comp_type]['embeddings'] = embeddings
            self.component_databases[comp_type]['paths'] = [str(p) for p in image_paths]
            self.component_databases[comp_type]['metadata'] = [
                {'component_type': comp_type, 'condition': 'damaged', 'path': str(p)}
                for p in image_paths
            ]
        
        self.is_ready = True
        logger.info("✅ RAG databases for Arcelik-Beko washing machines built successfully!")
        
        # Log statistics
        for comp_type, db in self.component_databases.items():
            logger.info(f"  {comp_type}: {len(db['embeddings'])} damaged reference images")
    
    def detect_component_type(self, image: Image.Image, top_k: int = 20) -> Dict:
        """
        Enhanced component type detection with better discrimination
        Uses multiple scoring methods to reduce confusion between similar components
        
        Args:
            image: Input image
            top_k: Number of similar images to check per component (default 20 for robustness)
            
        Returns:
            Dictionary with component detection results
        """
        if not self.is_ready:
            raise RuntimeError("Inspector not ready. Call build_component_databases() first.")
        
        # Get image embedding
        query_embedding = self.embedder.encode_image(image)
        
        # Check similarity against each component database
        component_scores = {}
        component_details = {}
        
        for comp_type, db in self.component_databases.items():
            if len(db['embeddings']) == 0:
                continue
            
            # Compute similarities to all images of this component type
            similarities = np.dot(db['embeddings'], query_embedding)
            
            # Get top-k matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_similarities = similarities[top_indices]
            
            # ENHANCED SCORING: Combine multiple metrics for better discrimination
            
            # 1. Weighted score (emphasizes top matches)
            weights = np.linspace(1.0, 0.3, len(top_similarities))  # Stronger emphasis on top matches
            weighted_score = float(np.sum(top_similarities * weights) / np.sum(weights))
            
            # 2. Max similarity (best single match)
            max_similarity = float(np.max(top_similarities))
            
            # 3. Top-5 average (consistency of top matches)
            top5_avg = float(np.mean(top_similarities[:min(5, len(top_similarities))]))
            
            # 4. Standard average
            avg_similarity = float(np.mean(top_similarities))
            
            # 5. Consistency score (lower std = more consistent matches)
            consistency = 1.0 - min(float(np.std(top_similarities[:10])), 0.3) / 0.3
            
            # HYBRID SCORE: Combine max, weighted, and consistency
            # Max similarity has highest weight to capture best match
            # Weighted score provides robustness
            # Consistency rewards clear component identity
            hybrid_score = (0.50 * max_similarity +    # Best match is most important
                          0.35 * weighted_score +     # Weighted average for robustness
                          0.15 * consistency)         # Consistency bonus
            
            component_scores[comp_type] = hybrid_score
            component_details[comp_type] = {
                'hybrid_score': hybrid_score,
                'weighted_score': weighted_score,
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'top5_avg': top5_avg,
                'consistency': consistency,
                'top_similarities': [float(s) for s in top_similarities],
                'num_references': len(db['embeddings'])
            }
        
        # Determine detected component with ENHANCED margin requirements
        if component_scores:
            sorted_components = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
            detected_component = sorted_components[0][0]
            confidence = sorted_components[0][1]
            
            # STRICTER margin requirements to reduce confusion
            if len(sorted_components) > 1:
                margin = sorted_components[0][1] - sorted_components[1][1]
                margin_ratio = margin / sorted_components[0][1] if sorted_components[0][1] > 0 else 0
                
                # Get max similarities for additional check
                max_sim_winner = component_details[sorted_components[0][0]]['max_similarity']
                max_sim_runner = component_details[sorted_components[1][0]]['max_similarity']
                max_sim_diff = max_sim_winner - max_sim_runner
                
                # MULTI-LEVEL DECISION LOGIC
                # Level 1: Clear winner (margin > 8% OR margin_ratio > 12%)
                if margin >= 0.08 or margin_ratio >= 0.12:
                    logger.info(
                        f"✅ Clear component: {sorted_components[0][0]} ({sorted_components[0][1]:.1%}), "
                        f"margin: {margin:.1%} ({margin_ratio:.1%} ratio)"
                    )
                
                # Level 2: Check max similarity difference as tiebreaker
                elif max_sim_diff >= 0.10:
                    # Winner has significantly better best match
                    logger.info(
                        f"✅ Component (by best match): {sorted_components[0][0]} "
                        f"(max_sim: {max_sim_winner:.1%} vs {max_sim_runner:.1%})"
                    )
                
                # Level 3: Uncertain - margin too small
                elif margin < 0.05:
                    logger.warning(
                        f"⚠️ UNCERTAIN component detection: {sorted_components[0][0]} ({sorted_components[0][1]:.1%}) "
                        f"vs {sorted_components[1][0]} ({sorted_components[1][1]:.1%}), margin: {margin:.1%}"
                    )
                    # Still return best guess but flag uncertainty
                    component_details['uncertainty_flag'] = True
                    component_details['uncertain_reason'] = (
                        f"Close scores: {sorted_components[0][0]}={sorted_components[0][1]:.1%} vs "
                        f"{sorted_components[1][0]}={sorted_components[1][1]:.1%}"
                    )
                
                # Level 4: Medium confidence
                else:
                    logger.info(
                        f"ℹ️ Component (medium confidence): {sorted_components[0][0]} ({sorted_components[0][1]:.1%}), "
                        f"margin: {margin:.1%}"
                    )
            
            # Additional check: very low confidence overall
            if confidence < 0.60:
                logger.warning(
                    f"⚠️ Low confidence component detection: {detected_component} ({confidence:.1%})"
                )
                component_details['low_confidence'] = True
        else:
            detected_component = 'unknown'
            confidence = 0.0
        
        return {
            'component_type': detected_component,
            'confidence': confidence,
            'all_scores': component_scores,
            'details': component_details
        }
    
    def refine_component_with_text(self, 
                                   image: Image.Image,
                                   visual_result: Dict) -> Dict:
        """
        Refine component detection using text-guided CLIP similarity
        This helps when visual similarity alone is ambiguous
        
        Args:
            image: Input image
            visual_result: Result from detect_component_type
            
        Returns:
            Refined component detection result
        """
        # Get image embedding
        query_embedding = self.embedder.encode_image(image)
        
        # Compute text-image similarity for each component type
        text_scores = {}
        for comp_type, description in self.COMPONENT_TEXT_DESCRIPTIONS.items():
            text_embedding = self.embedder.encode_text(description)
            similarity = float(np.dot(query_embedding, text_embedding))
            text_scores[comp_type] = similarity
        
        # Get visual scores
        visual_scores = visual_result['all_scores']
        
        # Check the margin between top 2 visual scores
        sorted_visual = sorted(visual_scores.items(), key=lambda x: x[1], reverse=True)
        visual_margin = sorted_visual[0][1] - sorted_visual[1][1] if len(sorted_visual) >= 2 else 1.0
        
        # ADAPTIVE WEIGHTING: Use more text when visual margin is very small
        # Very small margin (< 2%) → use 40/60 split (favor text as tiebreaker)
        # Small margin (2-5%) → use 50/50 split
        # Medium margin (5-10%) → use 65/35 split
        # Large margin (> 10%) → use 75/25 split (trust visual more)
        if visual_margin < 0.02:
            visual_weight, text_weight = 0.40, 0.60  # Favor text for very close visual scores
            logger.info(f"📊 Very small visual margin ({visual_margin:.1%}), using 40/60 visual/text (favoring text)")
        elif visual_margin < 0.05:
            visual_weight, text_weight = 0.50, 0.50  # Equal weight
            logger.info(f"📊 Small visual margin ({visual_margin:.1%}), using 50/50 visual/text")
        elif visual_margin < 0.10:
            visual_weight, text_weight = 0.65, 0.35  # Slight visual preference
            logger.info(f"📊 Medium visual margin ({visual_margin:.1%}), using 65/35 visual/text")
        else:
            visual_weight, text_weight = 0.75, 0.25  # Strong visual preference
        
        # Combine visual and text scores with adaptive weighting
        combined_scores = {}
        for comp_type in visual_scores.keys():
            visual_score = visual_scores[comp_type]
            text_score = text_scores.get(comp_type, 0.0)
            # Normalize text score to same range as visual
            text_score_norm = (text_score + 1.0) / 2.0  # CLIP text similarity is in [-1, 1]
            combined_scores[comp_type] = visual_weight * visual_score + text_weight * text_score_norm
        
        # Determine best component with combined scoring
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        refined_component = sorted_combined[0][0]
        refined_confidence = sorted_combined[0][1]
        
        # Check if text guidance changed the decision
        original_component = visual_result['component_type']
        if refined_component != original_component:
            logger.info(
                f"🔄 Text guidance changed detection: {original_component} → {refined_component} "
                f"(visual: {visual_scores[original_component]:.1%}, "
                f"text boost: {text_scores[refined_component]:.3f})"
            )
        
        return {
            'component_type': refined_component,
            'confidence': refined_confidence,
            'all_scores': combined_scores,
            'visual_scores': visual_scores,
            'text_scores': text_scores,
            'text_guided': refined_component != original_component,
            'original_detection': original_component,
            'details': visual_result['details']
        }
    
    def detect_component_type_enhanced(self, image: Image.Image, use_text_guidance: bool = True) -> Dict:
        """
        Enhanced component detection with optional text guidance
        
        Args:
            image: Input image
            use_text_guidance: Whether to use text-guided refinement
            
        Returns:
            Component detection results
        """
        # First, do visual-only detection
        visual_result = self.detect_component_type(image)
        
        # If ambiguous result or margin is small, use text guidance
        if use_text_guidance:
            # Check for ambiguity
            is_ambiguous = (
                visual_result['details'].get('uncertainty_flag', False) or
                visual_result['details'].get('low_confidence', False)
            )
            
            # Also check the margin between top 2 components
            all_scores = visual_result['all_scores']
            if len(all_scores) >= 2:
                sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
                margin = sorted_scores[0][1] - sorted_scores[1][1]
                # If margin is small (<10%), use text guidance
                is_ambiguous = is_ambiguous or margin < 0.10
            
            if is_ambiguous:
                logger.info("🔍 Ambiguous visual detection or small margin, applying text guidance...")
                return self.refine_component_with_text(image, visual_result)
        
        return visual_result
    
    def check_casting_damage_hybrid(self,
                                   image: Image.Image,
                                   top_k: int = 10) -> Dict:
        """
        Hybrid approach for casting damage detection:
        - Check similarity with GOOD casting examples (high similarity = good)
        - Check similarity with BROKEN casting examples (low similarity = additional confirmation it's good)
        - Use both to make confident decision
        
        Args:
            image: Input image
            top_k: Number of similar images to consider
            
        Returns:
            Dictionary with hybrid damage detection results
        """
        # Get image embedding
        query_embedding = self.embedder.encode_image(image)
        
        # Initialize results
        has_good_refs = (self.good_casting_database and 
                        len(self.good_casting_database['embeddings']) > 0)
        has_damaged_refs = ('casting' in self.component_databases and 
                           len(self.component_databases['casting']['embeddings']) > 0)
        
        if not has_good_refs and not has_damaged_refs:
            return {
                'is_damaged': False,
                'confidence': 0.0,
                'error': 'No casting references available (neither good nor damaged)',
                'decision': 'ERROR',
                'needs_review': True
            }
        
        # === Check similarity with GOOD casting examples ===
        good_similarity_max = 0.0
        good_similarity_avg = 0.0
        good_similar_images = []
        
        if has_good_refs:
            good_similarities = np.dot(self.good_casting_database['embeddings'], query_embedding)
            good_top_indices = np.argsort(good_similarities)[::-1][:top_k]
            good_top_similarities = good_similarities[good_top_indices]
            
            good_similarity_max = float(np.max(good_top_similarities))
            good_similarity_avg = float(np.mean(good_top_similarities))
            
            for idx in good_top_indices:
                good_similar_images.append({
                    'path': self.good_casting_database['paths'][idx],
                    'similarity': float(good_similarities[idx]),
                    'metadata': self.good_casting_database['metadata'][idx]
                })
        
        # === Check similarity with DAMAGED casting examples ===
        damaged_similarity_max = 0.0
        damaged_similarity_avg = 0.0
        damaged_similar_images = []
        
        if has_damaged_refs:
            damaged_similarities = np.dot(self.component_databases['casting']['embeddings'], query_embedding)
            damaged_top_indices = np.argsort(damaged_similarities)[::-1][:top_k]
            damaged_top_similarities = damaged_similarities[damaged_top_indices]
            
            damaged_similarity_max = float(np.max(damaged_top_similarities))
            damaged_similarity_avg = float(np.mean(damaged_top_similarities))
            
            for idx in damaged_top_indices:
                damaged_similar_images.append({
                    'path': self.component_databases['casting']['paths'][idx],
                    'similarity': float(damaged_similarities[idx]),
                    'metadata': self.component_databases['casting']['metadata'][idx]
                })
        
        # === HYBRID DECISION LOGIC ===
        uncertainty_signals = []
        needs_review = False
        
        # Case 1: We have BOTH good and damaged references (IDEAL)
        if has_good_refs and has_damaged_refs:
            # Calculate the difference between good and damaged similarity
            similarity_diff = good_similarity_max - damaged_similarity_max
            
            # High similarity to good + Clearly lower similarity to damaged = DEFINITELY GOOD
            if good_similarity_max >= 0.80 and damaged_similarity_max < 0.70:
                is_damaged = False
                confidence = good_similarity_max
                decision = 'GOOD (High confidence: matches good refs, differs from damaged refs)'
                logger.info(f"🧠 Hybrid: GOOD - good_sim={good_similarity_max:.1%}, damaged_sim={damaged_similarity_max:.1%}")
            
            # Good similarity is clearly higher than damaged (diff > 8%)
            elif good_similarity_max >= 0.70 and similarity_diff > 0.08:
                is_damaged = False
                confidence = good_similarity_max
                decision = 'GOOD (Matches good refs significantly better)'
                logger.info(f"🧠 Hybrid: GOOD - good_sim={good_similarity_max:.1%} > damaged_sim={damaged_similarity_max:.1%}")
            
            # Low similarity to good + High similarity to damaged = DAMAGED
            elif good_similarity_max < 0.70 and damaged_similarity_max >= 0.80:
                is_damaged = True
                confidence = damaged_similarity_max
                decision = 'DAMAGED (High confidence: matches damaged refs, differs from good refs)'
                logger.info(f"🧠 Hybrid: DAMAGED - good_sim={good_similarity_max:.1%}, damaged_sim={damaged_similarity_max:.1%}")
            
            # Damaged similarity is clearly higher than good (diff < -8%)
            elif damaged_similarity_max >= 0.75 and similarity_diff < -0.08:
                is_damaged = True
                confidence = damaged_similarity_max
                decision = 'DAMAGED (Matches damaged refs significantly better)'
                logger.info(f"🧠 Hybrid: DAMAGED - damaged_sim={damaged_similarity_max:.1%} > good_sim={good_similarity_max:.1%}")
            
            # High similarity to BOTH good and damaged = Check which is higher
            elif good_similarity_max >= 0.75 and damaged_similarity_max >= 0.75:
                # If difference is small (< 5%), it's uncertain
                if abs(similarity_diff) < 0.05:
                    is_damaged = False
                    confidence = 0.5
                    decision = 'UNCERTAIN (Very similar to both good and damaged refs)'
                    uncertainty_signals.append(f"High similarity to both good ({good_similarity_max:.1%}) and damaged ({damaged_similarity_max:.1%}), diff only {abs(similarity_diff):.1%}")
                    needs_review = True
                    logger.warning(f"⚠️ Hybrid: UNCERTAIN - both similarities high and very close")
                # Otherwise, go with whichever is higher (but flag for review)
                elif similarity_diff > 0:
                    is_damaged = False
                    confidence = good_similarity_max * 0.85  # Reduce confidence due to ambiguity
                    decision = f'PROBABLY GOOD (Slightly matches good refs better: {good_similarity_max:.1%} vs {damaged_similarity_max:.1%})'
                    uncertainty_signals.append(f"Close similarity: good={good_similarity_max:.1%}, damaged={damaged_similarity_max:.1%}, diff={similarity_diff:.1%}")
                    needs_review = True
                    logger.info(f"🧠 Hybrid: PROBABLY GOOD (close) - diff={similarity_diff:.1%}")
                else:
                    is_damaged = True
                    confidence = damaged_similarity_max * 0.85  # Reduce confidence due to ambiguity
                    decision = f'PROBABLY DAMAGED (Slightly matches damaged refs better: {damaged_similarity_max:.1%} vs {good_similarity_max:.1%})'
                    uncertainty_signals.append(f"Close similarity: good={good_similarity_max:.1%}, damaged={damaged_similarity_max:.1%}, diff={abs(similarity_diff):.1%}")
                    needs_review = True
                    logger.info(f"🧠 Hybrid: PROBABLY DAMAGED (close) - diff={abs(similarity_diff):.1%}")
            
            # Medium similarity to good, medium/low to damaged = PROBABLY GOOD
            elif good_similarity_max >= 0.70:
                is_damaged = False
                confidence = good_similarity_max
                decision = 'GOOD (Matches good refs better)'
                if damaged_similarity_max >= 0.65:
                    uncertainty_signals.append(f"Medium confidence: good_sim={good_similarity_max:.1%}, damaged_sim={damaged_similarity_max:.1%}")
                    needs_review = True
            
            # Medium similarity to damaged, low to good = PROBABLY DAMAGED
            elif damaged_similarity_max >= 0.70:
                is_damaged = True
                confidence = damaged_similarity_max
                decision = 'DAMAGED (Matches damaged refs better)'
                if good_similarity_max >= 0.60:
                    uncertainty_signals.append(f"Medium confidence: good_sim={good_similarity_max:.1%}, damaged_sim={damaged_similarity_max:.1%}")
                    needs_review = True
            
            # Low similarity to both = UNCERTAIN (might be different type or angle)
            else:
                is_damaged = False
                confidence = 0.3
                decision = 'UNCERTAIN (Low similarity to all references)'
                uncertainty_signals.append(f"Low similarity to both good ({good_similarity_max:.1%}) and damaged ({damaged_similarity_max:.1%})")
                needs_review = True
                logger.warning(f"⚠️ Hybrid: UNCERTAIN - both similarities low")
        
        # Case 2: Only GOOD references available
        elif has_good_refs:
            if good_similarity_max >= 0.80:
                is_damaged = False
                confidence = good_similarity_max
                decision = 'GOOD (High similarity to good refs)'
            elif good_similarity_max >= 0.65:
                is_damaged = False
                confidence = good_similarity_max
                decision = 'PROBABLY GOOD (Medium similarity to good refs)'
                uncertainty_signals.append(f"Medium similarity to good refs ({good_similarity_max:.1%})")
                needs_review = True
            else:
                is_damaged = True
                confidence = 1.0 - good_similarity_max
                decision = 'POSSIBLY DAMAGED (Low similarity to good refs - but no damaged refs to confirm)'
                uncertainty_signals.append(f"Low similarity to good refs ({good_similarity_max:.1%}), no damaged refs")
                needs_review = True
        
        # Case 3: Only DAMAGED references available (original behavior with smart override)
        else:
            if damaged_similarity_max >= 0.85:
                is_damaged = True
                confidence = damaged_similarity_max
                decision = 'DAMAGED (High similarity to damaged refs)'
            elif damaged_similarity_max < 0.70:
                # Smart override: low similarity to damaged = probably good
                is_damaged = False
                confidence = 1.0 - damaged_similarity_max
                decision = 'GOOD (Smart detection: low similarity to damaged refs)'
                uncertainty_signals.append(f"Only damaged refs available, low similarity ({damaged_similarity_max:.1%})")
                needs_review = True
                logger.info(f"🧠 Smart override: GOOD - low similarity to damaged refs")
            else:
                # Medium similarity - uncertain
                is_damaged = False
                confidence = 0.5
                decision = 'UNCERTAIN (Medium similarity to damaged refs, no good refs available)'
                uncertainty_signals.append(f"Only damaged refs available, medium similarity ({damaged_similarity_max:.1%})")
                needs_review = True
        
        # Build uncertainty reason
        uncertainty_reason = "; ".join(uncertainty_signals) if uncertainty_signals else "High confidence hybrid prediction"
        
        return {
            'is_damaged': is_damaged,
            'confidence': confidence,
            'decision': decision,
            'needs_review': needs_review,
            'uncertainty_reason': uncertainty_reason,
            'uncertainty_signals': uncertainty_signals,
            'num_signals': len(uncertainty_signals),
            # Good casting metrics
            'good_similarity_max': good_similarity_max,
            'good_similarity_avg': good_similarity_avg,
            'good_similar_images': good_similar_images,
            'has_good_refs': has_good_refs,
            # Damaged casting metrics
            'damaged_similarity_max': damaged_similarity_max,
            'damaged_similarity_avg': damaged_similarity_avg,
            'damaged_similar_images': damaged_similar_images,
            'has_damaged_refs': has_damaged_refs,
            # Hybrid approach indicator
            'hybrid_approach': True
        }
    
    def check_damage_rag(self, 
                        image: Image.Image, 
                        component_type: str,
                        top_k: int = 10,
                        damage_threshold: float = 0.75) -> Dict:
        """
        Check for damage using RAG similarity to damaged references
        
        Args:
            image: Input image
            component_type: Component type (pulley/casting/cover)
            top_k: Number of similar images to consider
            damage_threshold: Similarity threshold for damage detection
            
        Returns:
            Dictionary with damage detection results
        """
        if component_type not in self.component_databases:
            return {
                'is_damaged': False,
                'confidence': 0.0,
                'error': f'Unknown component type: {component_type}'
            }
        
        db = self.component_databases[component_type]
        
        if len(db['embeddings']) == 0:
            return {
                'is_damaged': False,
                'confidence': 0.0,
                'error': f'No references for {component_type}'
            }
        
        # Get image embedding
        query_embedding = self.embedder.encode_image(image)
        
        # Find similar damaged images
        similarities = np.dot(db['embeddings'], query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        # Get similar images info with weighted condition voting
        similar_images = []
        condition_votes = {}
        
        for idx in top_indices:
            metadata = db['metadata'][idx]
            similarity_score = float(similarities[idx])
            similar_images.append({
                'path': db['paths'][idx],
                'similarity': similarity_score,
                'metadata': metadata
            })
            # Weighted voting: weight by similarity score (smart classifier approach)
            condition = metadata.get('condition', 'damaged')
            condition_votes[condition] = condition_votes.get(condition, 0) + similarity_score
        
        # Calculate similarities
        avg_similarity = float(np.mean(top_similarities))
        max_similarity = float(np.max(top_similarities))
        min_similarity = float(np.min(top_similarities))
        
        # Calculate variance (Signal 5: High variance detection)
        similarity_std = float(np.std(top_similarities))
        
        # ADVANCED SMART CLASSIFIER LOGIC: 5 Uncertainty Detection Signals
        uncertainty_signals = []
        needs_review = False
        uncertain_override = False
        
        # SIGNAL 1: Low top similarity (< 70%)
        if max_similarity < 0.70:
            uncertainty_signals.append(f"Low similarity score ({max_similarity:.1%})")
            needs_review = True
        
        # SIGNAL 2: Close competition between conditions
        if len(condition_votes) > 1:
            sorted_votes = sorted(condition_votes.values(), reverse=True)
            if len(sorted_votes) >= 2:
                vote_diff = sorted_votes[0] - sorted_votes[1]
                vote_ratio = vote_diff / sorted_votes[0] if sorted_votes[0] > 0 else 0
                
                if vote_ratio < 0.15:  # Less than 15% difference
                    uncertainty_signals.append(
                        f"Close competition: damaged vs OK votes differ by only {vote_ratio:.1%}"
                    )
                    needs_review = True
        
        # SIGNAL 3: CRITICAL - Only damaged examples but low similarity (MOST IMPORTANT!)
        has_only_damaged = 'ok' not in condition_votes and 'good' not in condition_votes
        
        if has_only_damaged and max_similarity < 0.90:
            uncertainty_signals.append(
                f"⚠️ CRITICAL: Only damaged refs in database, but similarity is only {max_similarity:.1%}"
            )
            uncertain_override = True  # Override to mark as OK
            needs_review = True
        
        # SIGNAL 4: High variance in top similarities (unstable prediction)
        if similarity_std > 0.10:
            uncertainty_signals.append(
                f"High variance in similarities (std: {similarity_std:.3f})"
            )
            needs_review = True
        
        # SIGNAL 5: Medium confidence zone (70-85%)
        if 0.70 <= avg_similarity < 0.85 and not uncertain_override:
            uncertainty_signals.append(
                f"Medium confidence zone ({avg_similarity:.1%})"
            )
            needs_review = True
        
        # DECISION LOGIC with smart classification
        if uncertain_override:
            # CRITICAL OVERRIDE: Only damaged refs exist but low similarity → component is OK!
            is_damaged = False
            confidence = 1.0 - max_similarity  # Invert: low similarity to damaged = high confidence it's OK
            decision = 'GOOD (Smart detection: low similarity to damaged refs)'
            logger.info(f"🧠 Smart classifier override: Component is OK despite only damaged refs")
        elif avg_similarity >= damage_threshold:
            # High similarity to damaged → damaged
            is_damaged = True
            confidence = avg_similarity
            decision = 'DAMAGED'
            
            # Additional check: if confidence is borderline, flag for review
            if avg_similarity < 0.85:
                needs_review = True
        else:
            # Low similarity to damaged → good
            is_damaged = False
            confidence = 1.0 - avg_similarity  # Confidence it's good
            decision = 'GOOD'
        
        # Build comprehensive uncertainty reason
        uncertainty_reason = "; ".join(uncertainty_signals) if uncertainty_signals else "High confidence prediction"
        
        # Calculate weighted condition confidence
        total_votes = sum(condition_votes.values())
        condition_confidence = {}
        for cond, votes in condition_votes.items():
            condition_confidence[cond] = votes / total_votes if total_votes > 0 else 0
        
        return {
            'is_damaged': is_damaged,
            'confidence': confidence,
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity,
            'min_similarity': min_similarity,
            'similarity_std': similarity_std,
            'similar_images': similar_images,
            'threshold': damage_threshold,
            'decision': decision,
            'needs_review': needs_review,
            'uncertainty_reason': uncertainty_reason,
            'uncertainty_signals': uncertainty_signals,
            'num_signals': len(uncertainty_signals),
            'condition_distribution': condition_votes,
            'condition_confidence': condition_confidence,
            'has_only_damaged_refs': has_only_damaged,
            'uncertain_override': uncertain_override
        }
    
    def inspect_component(self, 
                         image: Image.Image,
                         expected_step: Optional[int] = None,
                         use_text_guidance: bool = True) -> Dict:
        """
        Complete inspection: detect component type + check damage
        
        Args:
            image: Input image
            expected_step: Expected step (1=pulley, 2=casting, 3=cover)
            use_text_guidance: Whether to use text-guided component detection
            
        Returns:
            Complete inspection results
        """
        # Step 1: Detect component type (with optional text guidance)
        if use_text_guidance:
            component_result = self.detect_component_type_enhanced(image, use_text_guidance=True)
        else:
            component_result = self.detect_component_type(image)
        detected_component = component_result['component_type']
        
        # Step 2: Check if correct component for step
        is_correct_step = True
        expected_component = None
        
        if expected_step and expected_step in self.STEP_COMPONENTS:
            expected_component = self.STEP_COMPONENTS[expected_step]
            is_correct_step = (detected_component == expected_component)
        
        # Step 3: Check damage (only if correct component)
        if is_correct_step and detected_component != 'unknown':
            # Use hybrid detection for casting (uses both good and damaged refs)
            if detected_component == 'casting':
                damage_result = self.check_casting_damage_hybrid(image)
            else:
                # Use regular RAG for pulley and cover
                damage_result = self.check_damage_rag(image, detected_component)
        else:
            damage_result = {
                'is_damaged': False,
                'confidence': 0.0,
                'decision': 'N/A'
            }
        
        # Generate comprehensive message with smart detection info
        if not is_correct_step:
            message = (
                f"❌ Wrong Component! Expected {expected_component.upper()}, "
                f"but detected {detected_component.upper()}. "
                f"Please upload the correct component."
            )
            status = 'error'
        elif damage_result['is_damaged']:
            confidence_info = damage_result.get('confidence', 0)
            num_signals = damage_result.get('num_signals', 0)
            needs_review = damage_result.get('needs_review', False)
            hybrid_approach = damage_result.get('hybrid_approach', False)
            
            if hybrid_approach and detected_component == 'casting':
                # Show hybrid analysis for casting
                good_sim = damage_result.get('good_similarity_max', 0)
                damaged_sim = damage_result.get('damaged_similarity_max', 0)
                base_message = (
                    f"⚠️ DAMAGE DETECTED in {detected_component.upper()}!\n"
                    f"🔍 Hybrid Analysis: Matches damaged refs ({damaged_sim:.1%}) better than good refs ({good_sim:.1%})\n"
                    f"Confidence: {confidence_info:.1%}"
                )
            else:
                base_message = (
                    f"⚠️ DAMAGE DETECTED in {detected_component.upper()}! "
                    f"Confidence: {confidence_info:.1%}"
                )
            
            if needs_review and num_signals > 0:
                message = base_message + f"\n⚠️ {num_signals} uncertainty signal(s) detected - recommend manual inspection"
            else:
                message = base_message
            status = 'damaged'
        else:
            # Component is GOOD
            confidence_info = damage_result.get('confidence', 0)
            uncertain_override = damage_result.get('uncertain_override', False)
            num_signals = damage_result.get('num_signals', 0)
            hybrid_approach = damage_result.get('hybrid_approach', False)
            
            if uncertain_override:
                # Smart classifier detected it's OK despite only having damaged refs
                message = (
                    f"✅ {detected_component.upper()} looks GOOD!\n"
                    f"🧠 Smart Detection: Low similarity to damaged references ({1-confidence_info:.1%})\n"
                    f"Confidence: {confidence_info:.1%}\n"
                    f"💡 Only damaged examples in database, but your component doesn't match them!"
                )
            elif hybrid_approach and detected_component == 'casting':
                # Hybrid approach was used for casting
                good_sim = damage_result.get('good_similarity_max', 0)
                damaged_sim = damage_result.get('damaged_similarity_max', 0)
                message = (
                    f"✅ {detected_component.upper()} looks GOOD!\n"
                    f"🔍 Hybrid Analysis: Matches good refs ({good_sim:.1%}) better than damaged refs ({damaged_sim:.1%})\n"
                    f"Confidence: {confidence_info:.1%}"
                )
                if num_signals > 0:
                    message += f"\n⚠️ Note: {num_signals} uncertainty signal(s) detected"
            elif num_signals > 0:
                message = (
                    f"✅ {detected_component.upper()} looks GOOD! "
                    f"Confidence: {confidence_info:.1%}\n"
                    f"⚠️ Note: {num_signals} uncertainty signal(s) detected"
                )
            else:
                message = (
                    f"✅ {detected_component.upper()} looks GOOD! "
                    f"Confidence: {confidence_info:.1%} (High confidence)"
                )
            
            status = 'good'
        
        return {
            'component_type': detected_component,
            'component_confidence': component_result['confidence'],
            'component_scores': component_result['all_scores'],
            'is_correct_step': is_correct_step,
            'expected_component': expected_component,
            'step': expected_step,
            'is_damaged': damage_result['is_damaged'],
            'damage_confidence': damage_result['confidence'],
            'damage_details': damage_result,
            'message': message,
            'status': status,
            'component_details': component_result['details']
        }
    
    def save_databases(self, save_path: str) -> None:
        """Save component databases"""
        import pickle
        
        data = {
            'component_databases': self.component_databases,
            'good_casting_database': self.good_casting_database,
            'is_ready': self.is_ready
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"💾 Saved databases to {save_path}")
    
    def load_databases(self, load_path: str) -> None:
        """Load component databases"""
        import pickle
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.component_databases = data['component_databases']
        self.good_casting_database = data.get('good_casting_database', None)
        self.is_ready = data['is_ready']
        
        # Log what was loaded
        logger.info(f"📂 Loaded databases from {load_path}")
        for comp_type, db in self.component_databases.items():
            logger.info(f"  {comp_type}: {len(db['embeddings'])} damaged references")
        
        if self.good_casting_database and len(self.good_casting_database.get('embeddings', [])) > 0:
            logger.info(f"  casting (GOOD): {len(self.good_casting_database['embeddings'])} references")
        else:
            logger.warning("  ⚠️ No good casting references in loaded database")