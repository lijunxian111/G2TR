import torch
from torch import nn
import torch.nn.functional as F
import math

def token_merging(image_embeds, keep_indices, scaling=1):
    """
    Merges non-retained tokens with their nearest retained tokens based on cosine similarity.

    Args:
        image_embeds (Tensor): Tensor of shape (N, D) where N is the number of tokens, and D is the feature dimension.
        keep_indices (Tensor): Tensor of shape (T, ), where T is the number of retained tokens.

    Returns:
        merged_features (Tensor): Tensor of shape (T, D) where T is the number of retained tokens
                                and D is the feature dimension. The merged features are
                                the average of the retained token and the non-retained tokens.
    """
    N, D = image_embeds.shape
    T = len(keep_indices)
    
    keep_index_mask = torch.zeros(N, dtype=torch.bool, device=image_embeds.device)
    keep_index_mask[keep_indices] = True
    
    retained_tokens = image_embeds[keep_index_mask, :] # [T, D]
    non_retained_tokens = image_embeds[~keep_index_mask, :] # [N - T, D]
    # print(retained_tokens.shape, non_retained_tokens.shape)
    
    #print(retained_tokens.shape)
    non_retained_norm = F.normalize(non_retained_tokens, p=2, dim=-1)
    retained_norm = F.normalize(retained_tokens, p=2, dim=-1)

    cosine_sim = non_retained_norm @ retained_norm.T   # [N, M] to save memory
    #cosine_sim = F.cosine_similarity(non_retained_tokens.unsqueeze(1), retained_tokens.unsqueeze(0), dim=2)
    nearest_token_indices = cosine_sim.argmax(dim=1) # [N - T]
    # print(nearest_token_indices)
    
    merged_features = torch.zeros_like(retained_tokens) # [T, D]
    merged_features += retained_tokens * scaling
    
    expanded_indices = nearest_token_indices # [N - T]
    merged_features.scatter_add_(0, expanded_indices.unsqueeze(-1).expand(-1, D), non_retained_tokens)
    
    merge_count = torch.zeros(T, device=image_embeds.device, dtype=torch.int) # [T]
    merge_count.scatter_add_(0, expanded_indices, torch.ones_like(expanded_indices, dtype=merge_count.dtype))
    merged_features /= (scaling + merge_count.unsqueeze(1))
    
    return merged_features

def window_selection(attn_weights, num_keep_tokens, token_h, token_w, window_size=4):
    """
    Selects the top num_keep_tokens tokens with the highest attention weights. 
    The tokens are selected from non-overlapping windows of size window_size x window_size.
    Args:
        attn_weights (Tensor): Tensor of shape (N, ) where N is the number of tokens.
        num_keep_tokens (int): The number of tokens to keep.
        window_size (int): The size of the window to select the tokens from.
    """
    # start_time = time.time()
    
    #token_h, token_w = image_grid_thw[0, 1] // 2, image_grid_thw[0, 2] // 2
    assert token_h * token_w == attn_weights.shape[0], "The number of tokens in the window is not equal to num_keep_tokens"
    
    #num_windows_h = (token_h / window_size).floor().int()
    #num_windows_w = (token_w / window_size).floor().int()
    num_windows_h = math.floor(token_h / window_size)
    num_windows_w = math.floor(token_w / window_size)
    num_windows = num_windows_h * num_windows_w
    extra_h = token_h - (num_windows_h * window_size)
    extra_w = token_w - (num_windows_w * window_size)

    # attn_weights = attn_weights.view(token_h, token_w)
    # if extra_h > 0:
    #     attn_weights = attn_weights[:num_windows_h * window_size, :]
    # if extra_w > 0:
    #     attn_weights = attn_weights[:, :num_windows_w * window_size]
    # attn_weights = attn_weights.view(num_windows_h, window_size, num_windows_w, window_size)
    # attn_weights = attn_weights.permute(0, 2, 1, 3).reshape(num_windows_h * num_windows_w, window_size * window_size)

    sorted_indices = torch.argsort(attn_weights, dim=0, descending=True)
    window_counter = torch.zeros(token_h, token_w, device=attn_weights.device, dtype=torch.int)
    total_counter = 0
    
    # Calculate the limit of the number of tokens to keep in each window
    #limit = (num_keep_tokens / num_windows).ceil().int()
    limit = math.ceil(num_keep_tokens / num_windows)
    

    keep_indices = torch.zeros(num_keep_tokens, device=attn_weights.device, dtype=torch.int)
    for index in sorted_indices:
        x = (index // token_w) // window_size
        y = (index % token_w) // window_size
        if x == num_windows_h:
            x = num_windows_h - 1
        if y == num_windows_w:
            y = num_windows_w - 1
        #x = x.int()
        #y = y.int()
        if window_counter[x, y] < limit:
            window_counter[x, y] += 1
            keep_indices[total_counter] = index
            total_counter += 1
        if total_counter >= num_keep_tokens:
            break
        
    # end_time = time.time()
    # print(f"Window selection time: {end_time - start_time:.4f} seconds")

    assert total_counter == num_keep_tokens, "The number of tokens to keep is not equal to num_keep_tokens"
    return keep_indices

def reduce_attn_to_token_score(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    attn_weights: [num_heads, seq_len, seq_len]
    return: [seq_len]
    """
    # sum across heads -> [seq_len, seq_len]
    attn_weights = torch.sum(attn_weights, dim=0)
    # mean across queries -> [seq_len]
    attn_weights = torch.mean(attn_weights, dim=0)
    return attn_weights

def window_selection_packed(attn_weights, num_keep_tokens, cu_seqlens, max_seqlen, window_size=4, image_shapes=None):
    """
    Packed version of window selection.
    Each segment in cu_seqlens corresponds to one image.

    Args:
        attn_weights: Tensor [N]
        num_keep_tokens: int, total keep tokens across all packed images
        cu_seqlens: IntTensor [B+1]
        max_seqlen: int
        window_size: int

    Returns:
        keep_indices: LongTensor [num_keep_tokens]
    """
    device = attn_weights.device
    total_tokens = attn_weights.shape[0]
    num_images = cu_seqlens.numel() - 1

    assert total_tokens == cu_seqlens[-1].item(), \
        f"attn_weights length {total_tokens} != cu_seqlens[-1] {cu_seqlens[-1].item()}"

    # allocate keep budget proportional to each image length
    image_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    total_len = sum(image_lens)

    keep_indices_all = []
    assigned = 0
    #print(num_images)
    #print(cu_seqlens.shape)
    for img_idx in range(num_images):
        s = cu_seqlens[img_idx].item()
        e = cu_seqlens[img_idx + 1].item()
        cur_len = e - s
        cur_scores = attn_weights[s:e]
        #print(s, e, cur_len)
        # assume square token grid
        if image_shapes is not None:
            shape = image_shapes[img_idx]
            side_h = shape[1] 
            side_w = shape[2] 
            assert side_h * side_w == cur_len, \
                f"image {img_idx} token length {cur_len} does not match image shape {shape}, cannot infer token_h/token_w"
        else:
            side_w = side_h = int(round(math.sqrt(cur_len)))
            assert side_w * side_h == cur_len, \
                f"image {img_idx} token length {cur_len} is not square, cannot infer token_h/token_w"

        # proportional budget; make sure last image absorbs rounding residue
        if img_idx < num_images - 1:
            cur_keep = max(1, round(num_keep_tokens * cur_len / total_len))
            assigned += cur_keep
        else:
            cur_keep = num_keep_tokens - assigned
            cur_keep = max(1, cur_keep)

        cur_keep = min(cur_keep, cur_len)

        cur_keep_indices = window_selection(
            cur_scores,
            num_keep_tokens=cur_keep,
            token_h=side_h,
            token_w=side_w,
            window_size=window_size,
        )
        keep_indices_all.append(cur_keep_indices + s)

    keep_indices = torch.cat(keep_indices_all, dim=0)
    keep_indices = torch.sort(torch.unique(keep_indices)).values

    # 若因 round 导致数量不精确，则补/裁到目标个数
    if keep_indices.numel() > num_keep_tokens:
        keep_scores = attn_weights[keep_indices]
        keep_indices = keep_indices[torch.topk(keep_scores, k=num_keep_tokens).indices]
        keep_indices = torch.sort(keep_indices).values
    elif keep_indices.numel() < num_keep_tokens:
        all_idx = torch.arange(total_tokens, device=device)
        mask = torch.zeros(total_tokens, device=device, dtype=torch.bool)
        mask[keep_indices] = True
        remain_idx = all_idx[~mask]
        remain_scores = attn_weights[remain_idx]
        extra_k = num_keep_tokens - keep_indices.numel()
        extra_idx = remain_idx[torch.topk(remain_scores, k=extra_k).indices]
        keep_indices = torch.cat([keep_indices, extra_idx], dim=0)
        keep_indices = torch.sort(torch.unique(keep_indices)).values

    assert keep_indices.numel() == num_keep_tokens, \
        f"final keep count {keep_indices.numel()} != expected {num_keep_tokens}"

    return keep_indices

def rebuild_packed_vit_after_prune(
    packed_flattened_position_ids: torch.LongTensor,
    cu_seqlens: torch.IntTensor,
    keep_indices: torch.LongTensor,
):
    """
    根据全局 keep_indices，重建：
      - kept_position_ids
      - new_cu_seqlens

    Args:
        packed_flattened_position_ids: [N]
        cu_seqlens: [B+1]
        keep_indices: [K], global indices on packed tokens

    Returns:
        kept_position_ids: [K]
        new_cu_seqlens: [B+1]
    """
    device = packed_flattened_position_ids.device
    keep_indices = torch.sort(torch.unique(keep_indices)).values.to(device)
    kept_position_ids = packed_flattened_position_ids[keep_indices]

    keep_mask = torch.zeros(cu_seqlens[-1].item(), device=device, dtype=torch.bool)
    keep_mask[keep_indices] = True

    new_lens = []
    for i in range(cu_seqlens.numel() - 1):
        s = cu_seqlens[i].item()
        e = cu_seqlens[i + 1].item()
        new_lens.append(int(keep_mask[s:e].sum().item()))

    new_cu_seqlens = [0]
    for l in new_lens:
        new_cu_seqlens.append(new_cu_seqlens[-1] + l)
    new_cu_seqlens = torch.tensor(new_cu_seqlens, device=device, dtype=cu_seqlens.dtype)

    return kept_position_ids, new_cu_seqlens
    


def latent_guided_keep_indices(
    vit_features: torch.Tensor,
    vit_cu_seqlens: torch.IntTensor,
    image_shapes,
    latent_anchor_features: torch.Tensor,
    latent_anchor_shapes,
    keep_ratio: float = 0.5,
    min_keep_tokens: int = 1,
):
    """
    Select keep indices for packed ViT tokens using a *single* latent-guided score.

    Score for each ViT token is the cosine similarity between the token and the
    latent anchor feature of its mapped latent cell.

    Args:
        vit_features: [N_vit, C]
        vit_cu_seqlens: [B+1]
        image_shapes: list[[C, h_vit, w_vit]]
        latent_anchor_features: [N_lat, C]
        latent_anchor_shapes: list[(h_lat, w_lat)]
        keep_ratio: keep ratio over ViT tokens of each image.
        min_keep_tokens: minimum keep tokens per image.

    Returns:
        keep_indices: global indices over packed ViT tokens.
        token_scores: [N_vit]
    """
    device = vit_features.device
    dtype = vit_features.dtype
    num_images = vit_cu_seqlens.numel() - 1

    assert len(image_shapes) == num_images
    assert len(latent_anchor_shapes) == num_images

    vit_norm = F.normalize(vit_features, dim=-1)
    all_scores = []
    keep_indices_all = []
    lat_start = 0

    for img_idx in range(num_images):
        v_s = vit_cu_seqlens[img_idx].item()
        v_e = vit_cu_seqlens[img_idx + 1].item()
        vit_seg = vit_norm[v_s:v_e]
        num_vit = v_e - v_s

        _, h_vit, w_vit = image_shapes[img_idx]
        h_lat, w_lat = latent_anchor_shapes[img_idx]
        num_lat = h_lat * w_lat

        assert h_vit * w_vit == num_vit, (
            f"image {img_idx}: vit len {num_vit} != h_vit*w_vit {h_vit*w_vit}"
        )

        lat_seg = latent_anchor_features[lat_start: lat_start + num_lat]
        lat_seg = F.normalize(lat_seg, dim=-1)
        lat_start += num_lat

        local_idx = torch.arange(num_vit, device=device)
        row = torch.div(local_idx, w_vit, rounding_mode='floor')
        col = local_idx % w_vit
        lat_row = torch.clamp(torch.div(row * h_lat, h_vit, rounding_mode='floor'), max=h_lat - 1)
        lat_col = torch.clamp(torch.div(col * w_lat, w_vit, rounding_mode='floor'), max=w_lat - 1)
        cell_id = lat_row * w_lat + lat_col

        token_scores = (vit_seg * lat_seg[cell_id]).sum(dim=-1)
        all_scores.append(token_scores)

        keep_n = max(min_keep_tokens, int(round(num_vit * keep_ratio)))
        keep_n = min(keep_n, num_vit)

        unique_cells = torch.unique(cell_id)
        cell_best_scores = []
        cell_best_token_idx = []
        for c in unique_cells.tolist():
            mask = cell_id == c
            score_c = token_scores.masked_fill(~mask, torch.finfo(dtype).min)
            best_idx = torch.argmax(score_c)
            cell_best_scores.append(token_scores[best_idx])
            cell_best_token_idx.append(best_idx)

        cell_best_scores = torch.stack(cell_best_scores)
        cell_best_token_idx = torch.stack(cell_best_token_idx)

        num_seed_cells = min(keep_n, unique_cells.numel())
        top_cells = torch.topk(cell_best_scores, k=num_seed_cells, largest=True).indices
        keep_local = cell_best_token_idx[top_cells]

        if keep_local.numel() < keep_n:
            keep_mask = torch.zeros(num_vit, device=device, dtype=torch.bool)
            keep_mask[keep_local] = True
            extra_scores = token_scores.masked_fill(keep_mask, torch.finfo(dtype).min)
            extra_k = keep_n - keep_local.numel()
            extra_local = torch.topk(extra_scores, k=extra_k, largest=True).indices
            keep_local = torch.cat([keep_local, extra_local], dim=0)

        keep_local = torch.sort(torch.unique(keep_local)).values
        if keep_local.numel() > keep_n:
            keep_local_scores = token_scores[keep_local]
            keep_local = keep_local[torch.topk(keep_local_scores, k=keep_n, largest=True).indices]
            keep_local = torch.sort(keep_local).values

        keep_indices_all.append(keep_local + v_s)

    keep_indices = torch.cat(keep_indices_all, dim=0)
    token_scores = torch.cat(all_scores, dim=0)
    return keep_indices, token_scores
