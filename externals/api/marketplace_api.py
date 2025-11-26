"""
Agent Marketplace API
-------------------
API endpoints for the agent marketplace, allowing users to share and rent agents.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field
from datetime import datetime
import json
import uuid
from pathlib import Path as FilePath

from .auth import User, get_current_active_user
from ..core.agent_builder import AgentBuilder
from ..utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()
agent_builder = AgentBuilder()

# ----- Pydantic Models -----

class AgentListing(BaseModel):
    id: str
    agent_id: str
    owner_id: str
    name: str
    description: str
    price_per_use: float
    price_per_month: Optional[float] = None
    capabilities: List[str]
    skills: List[str]
    rating: Optional[float] = None
    reviews_count: int = 0
    created_at: str
    updated_at: str
    is_public: bool = True
    tags: List[str] = []

class AgentListingCreate(BaseModel):
    agent_id: str
    name: str
    description: str
    price_per_use: float
    price_per_month: Optional[float] = None
    is_public: bool = True
    tags: List[str] = []

class AgentListingUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price_per_use: Optional[float] = None
    price_per_month: Optional[float] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None

class AgentReview(BaseModel):
    id: str
    listing_id: str
    user_id: str
    rating: float = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    created_at: str

class AgentReviewCreate(BaseModel):
    rating: float = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class AgentRental(BaseModel):
    id: str
    listing_id: str
    user_id: str
    rental_type: str  # 'per_use' or 'subscription'
    start_date: str
    end_date: Optional[str] = None
    status: str  # 'active', 'expired', 'cancelled'
    created_at: str
    updated_at: str

class AgentRentalCreate(BaseModel):
    listing_id: str
    rental_type: str  # 'per_use' or 'subscription'

# ----- Database Functions -----

# In a real application, these would be database operations
# For simplicity, we're using JSON files

MARKETPLACE_DIR = FilePath(__file__).parent.parent / "data" / "marketplace"
LISTINGS_FILE = MARKETPLACE_DIR / "listings.json"
REVIEWS_FILE = MARKETPLACE_DIR / "reviews.json"
RENTALS_FILE = MARKETPLACE_DIR / "rentals.json"

def _ensure_marketplace_files():
    """Ensure marketplace files exist"""
    MARKETPLACE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not LISTINGS_FILE.exists():
        with open(LISTINGS_FILE, "w") as f:
            json.dump({}, f)
    
    if not REVIEWS_FILE.exists():
        with open(REVIEWS_FILE, "w") as f:
            json.dump({}, f)
    
    if not RENTALS_FILE.exists():
        with open(RENTALS_FILE, "w") as f:
            json.dump({}, f)

def _get_listings() -> Dict[str, Dict[str, Any]]:
    """Get all agent listings"""
    _ensure_marketplace_files()
    
    with open(LISTINGS_FILE, "r") as f:
        return json.load(f)

def _save_listings(listings: Dict[str, Dict[str, Any]]):
    """Save agent listings"""
    with open(LISTINGS_FILE, "w") as f:
        json.dump(listings, f, indent=2)

def _get_reviews() -> Dict[str, Dict[str, Any]]:
    """Get all agent reviews"""
    _ensure_marketplace_files()
    
    with open(REVIEWS_FILE, "r") as f:
        return json.load(f)

def _save_reviews(reviews: Dict[str, Dict[str, Any]]):
    """Save agent reviews"""
    with open(REVIEWS_FILE, "w") as f:
        json.dump(reviews, f, indent=2)

def _get_rentals() -> Dict[str, Dict[str, Any]]:
    """Get all agent rentals"""
    _ensure_marketplace_files()
    
    with open(RENTALS_FILE, "r") as f:
        return json.load(f)

def _save_rentals(rentals: Dict[str, Dict[str, Any]]):
    """Save agent rentals"""
    with open(RENTALS_FILE, "w") as f:
        json.dump(rentals, f, indent=2)

def _update_listing_rating(listing_id: str):
    """Update a listing's rating based on reviews"""
    listings = _get_listings()
    reviews = _get_reviews()
    
    if listing_id not in listings:
        return
    
    # Get all reviews for this listing
    listing_reviews = [
        review for review in reviews.values()
        if review["listing_id"] == listing_id
    ]
    
    if not listing_reviews:
        listings[listing_id]["rating"] = None
        listings[listing_id]["reviews_count"] = 0
    else:
        # Calculate average rating
        total_rating = sum(review["rating"] for review in listing_reviews)
        avg_rating = total_rating / len(listing_reviews)
        
        listings[listing_id]["rating"] = round(avg_rating, 1)
        listings[listing_id]["reviews_count"] = len(listing_reviews)
    
    _save_listings(listings)

# ----- API Endpoints -----

@router.post("/marketplace/listings", response_model=AgentListing)
async def create_agent_listing(listing_data: AgentListingCreate, current_user: User = Depends(get_current_active_user)):
    """Create a new agent listing in the marketplace"""
    try:
        # Check if agent exists
        agent = agent_builder.get_agent(listing_data.agent_id)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {listing_data.agent_id} not found"
            )
        
        # Check if user owns the agent
        # In a real application, you would check agent ownership
        
        # Create listing
        listing_id = f"listing_{str(uuid.uuid4())}"
        now = datetime.now().isoformat()
        
        listing = {
            "id": listing_id,
            "agent_id": listing_data.agent_id,
            "owner_id": current_user.id,
            "name": listing_data.name,
            "description": listing_data.description,
            "price_per_use": listing_data.price_per_use,
            "price_per_month": listing_data.price_per_month,
            "capabilities": agent.capabilities,
            "skills": agent.skills if hasattr(agent, 'skills') else [],
            "rating": None,
            "reviews_count": 0,
            "created_at": now,
            "updated_at": now,
            "is_public": listing_data.is_public,
            "tags": listing_data.tags
        }
        
        # Save listing
        listings = _get_listings()
        listings[listing_id] = listing
        _save_listings(listings)
        
        return listing
    except Exception as e:
        logger.error(f"Error creating agent listing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating agent listing: {str(e)}"
        )

@router.get("/marketplace/listings", response_model=List[AgentListing])
async def get_agent_listings(
    search: Optional[str] = None,
    tags: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None,
    owner_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get agent listings with optional filtering"""
    try:
        listings = _get_listings()
        
        # Convert to list
        listings_list = list(listings.values())
        
        # Apply filters
        if search:
            search = search.lower()
            listings_list = [
                listing for listing in listings_list
                if search in listing["name"].lower() or search in listing["description"].lower()
            ]
        
        if tags:
            tag_list = tags.split(",")
            listings_list = [
                listing for listing in listings_list
                if any(tag in listing["tags"] for tag in tag_list)
            ]
        
        if min_rating is not None:
            listings_list = [
                listing for listing in listings_list
                if listing["rating"] is not None and listing["rating"] >= min_rating
            ]
        
        if max_price is not None:
            listings_list = [
                listing for listing in listings_list
                if listing["price_per_use"] <= max_price
            ]
        
        if owner_id:
            listings_list = [
                listing for listing in listings_list
                if listing["owner_id"] == owner_id
            ]
        
        # Only show public listings unless user is owner
        listings_list = [
            listing for listing in listings_list
            if listing["is_public"] or listing["owner_id"] == current_user.id
        ]
        
        return listings_list
    except Exception as e:
        logger.error(f"Error getting agent listings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting agent listings: {str(e)}"
        )

@router.get("/marketplace/listings/{listing_id}", response_model=AgentListing)
async def get_agent_listing(listing_id: str, current_user: User = Depends(get_current_active_user)):
    """Get a specific agent listing"""
    try:
        listings = _get_listings()
        
        if listing_id not in listings:
            raise HTTPException(
                status_code=404,
                detail=f"Listing {listing_id} not found"
            )
        
        listing = listings[listing_id]
        
        # Check if listing is public or user is owner
        if not listing["is_public"] and listing["owner_id"] != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to view this listing"
            )
        
        return listing
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent listing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting agent listing: {str(e)}"
        )

@router.put("/marketplace/listings/{listing_id}", response_model=AgentListing)
async def update_agent_listing(
    listing_id: str,
    listing_data: AgentListingUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update an agent listing"""
    try:
        listings = _get_listings()
        
        if listing_id not in listings:
            raise HTTPException(
                status_code=404,
                detail=f"Listing {listing_id} not found"
            )
        
        listing = listings[listing_id]
        
        # Check if user is owner
        if listing["owner_id"] != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to update this listing"
            )
        
        # Update fields
        update_data = listing_data.dict(exclude_unset=True)
        
        for key, value in update_data.items():
            listing[key] = value
        
        listing["updated_at"] = datetime.now().isoformat()
        
        # Save listing
        listings[listing_id] = listing
        _save_listings(listings)
        
        return listing
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent listing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating agent listing: {str(e)}"
        )

@router.delete("/marketplace/listings/{listing_id}")
async def delete_agent_listing(listing_id: str, current_user: User = Depends(get_current_active_user)):
    """Delete an agent listing"""
    try:
        listings = _get_listings()
        
        if listing_id not in listings:
            raise HTTPException(
                status_code=404,
                detail=f"Listing {listing_id} not found"
            )
        
        listing = listings[listing_id]
        
        # Check if user is owner
        if listing["owner_id"] != current_user.id and not current_user.is_admin:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to delete this listing"
            )
        
        # Delete listing
        del listings[listing_id]
        _save_listings(listings)
        
        return {"status": "success", "message": f"Listing {listing_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent listing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting agent listing: {str(e)}"
        )

@router.post("/marketplace/listings/{listing_id}/reviews", response_model=AgentReview)
async def create_agent_review(
    listing_id: str,
    review_data: AgentReviewCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a review for an agent listing"""
    try:
        listings = _get_listings()
        
        if listing_id not in listings:
            raise HTTPException(
                status_code=404,
                detail=f"Listing {listing_id} not found"
            )
        
        # Check if user has rented this agent
        rentals = _get_rentals()
        user_rentals = [
            rental for rental in rentals.values()
            if rental["user_id"] == current_user.id and rental["listing_id"] == listing_id
        ]
        
        if not user_rentals:
            raise HTTPException(
                status_code=403,
                detail="You must rent this agent before reviewing it"
            )
        
        # Check if user has already reviewed this listing
        reviews = _get_reviews()
        user_reviews = [
            review for review in reviews.values()
            if review["user_id"] == current_user.id and review["listing_id"] == listing_id
        ]
        
        if user_reviews:
            raise HTTPException(
                status_code=400,
                detail="You have already reviewed this listing"
            )
        
        # Create review
        review_id = f"review_{str(uuid.uuid4())}"
        now = datetime.now().isoformat()
        
        review = {
            "id": review_id,
            "listing_id": listing_id,
            "user_id": current_user.id,
            "rating": review_data.rating,
            "comment": review_data.comment,
            "created_at": now
        }
        
        # Save review
        reviews[review_id] = review
        _save_reviews(reviews)
        
        # Update listing rating
        _update_listing_rating(listing_id)
        
        return review
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating agent review: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating agent review: {str(e)}"
        )

@router.get("/marketplace/listings/{listing_id}/reviews", response_model=List[AgentReview])
async def get_agent_reviews(listing_id: str, current_user: User = Depends(get_current_active_user)):
    """Get reviews for an agent listing"""
    try:
        listings = _get_listings()
        
        if listing_id not in listings:
            raise HTTPException(
                status_code=404,
                detail=f"Listing {listing_id} not found"
            )
        
        # Get all reviews for this listing
        reviews = _get_reviews()
        listing_reviews = [
            review for review in reviews.values()
            if review["listing_id"] == listing_id
        ]
        
        return listing_reviews
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent reviews: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting agent reviews: {str(e)}"
        )

@router.post("/marketplace/rentals", response_model=AgentRental)
async def rent_agent(rental_data: AgentRentalCreate, current_user: User = Depends(get_current_active_user)):
    """Rent an agent from the marketplace"""
    try:
        listings = _get_listings()
        
        if rental_data.listing_id not in listings:
            raise HTTPException(
                status_code=404,
                detail=f"Listing {rental_data.listing_id} not found"
            )
        
        listing = listings[rental_data.listing_id]
        
        # Check if listing is public
        if not listing["is_public"]:
            raise HTTPException(
                status_code=403,
                detail="This listing is not available for rental"
            )
        
        # Check if user is not the owner
        if listing["owner_id"] == current_user.id:
            raise HTTPException(
                status_code=400,
                detail="You cannot rent your own agent"
            )
        
        # Create rental
        rental_id = f"rental_{str(uuid.uuid4())}"
        now = datetime.now().isoformat()
        
        rental = {
            "id": rental_id,
            "listing_id": rental_data.listing_id,
            "user_id": current_user.id,
            "rental_type": rental_data.rental_type,
            "start_date": now,
            "end_date": None,  # Will be set for subscriptions
            "status": "active",
            "created_at": now,
            "updated_at": now
        }
        
        # Set end date for subscriptions
        if rental_data.rental_type == "subscription":
            # Set end date to 30 days from now
            end_date = datetime.now() + timedelta(days=30)
            rental["end_date"] = end_date.isoformat()
        
        # Save rental
        rentals = _get_rentals()
        rentals[rental_id] = rental
        _save_rentals(rentals)
        
        return rental
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renting agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error renting agent: {str(e)}"
        )

@router.get("/marketplace/rentals", response_model=List[AgentRental])
async def get_user_rentals(current_user: User = Depends(get_current_active_user)):
    """Get all rentals for the current user"""
    try:
        rentals = _get_rentals()
        
        # Get all rentals for this user
        user_rentals = [
            rental for rental in rentals.values()
            if rental["user_id"] == current_user.id
        ]
        
        return user_rentals
    except Exception as e:
        logger.error(f"Error getting user rentals: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting user rentals: {str(e)}"
        )

@router.get("/marketplace/my-listings", response_model=List[AgentListing])
async def get_user_listings(current_user: User = Depends(get_current_active_user)):
    """Get all listings owned by the current user"""
    try:
        listings = _get_listings()
        
        # Get all listings for this user
        user_listings = [
            listing for listing in listings.values()
            if listing["owner_id"] == current_user.id
        ]
        
        return user_listings
    except Exception as e:
        logger.error(f"Error getting user listings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting user listings: {str(e)}"
        )

@router.get("/marketplace/stats")
async def get_marketplace_stats(current_user: User = Depends(get_current_active_user)):
    """Get marketplace statistics"""
    try:
        listings = _get_listings()
        reviews = _get_reviews()
        rentals = _get_rentals()
        
        # Calculate statistics
        total_listings = len(listings)
        total_reviews = len(reviews)
        total_rentals = len(rentals)
        
        # Get top rated listings
        listings_list = list(listings.values())
        top_rated = sorted(
            [l for l in listings_list if l["rating"] is not None],
            key=lambda x: x["rating"],
            reverse=True
        )[:5]
        
        # Get most rented listings
        rental_counts = {}
        for rental in rentals.values():
            listing_id = rental["listing_id"]
            rental_counts[listing_id] = rental_counts.get(listing_id, 0) + 1
        
        most_rented = sorted(
            [(listing_id, count) for listing_id, count in rental_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        most_rented_listings = [
            {**listings[listing_id], "rental_count": count}
            for listing_id, count in most_rented
            if listing_id in listings
        ]
        
        return {
            "total_listings": total_listings,
            "total_reviews": total_reviews,
            "total_rentals": total_rentals,
            "top_rated": top_rated,
            "most_rented": most_rented_listings
        }
    except Exception as e:
        logger.error(f"Error getting marketplace stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting marketplace stats: {str(e)}"
        )
