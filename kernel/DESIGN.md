# FusedLinearAttention Kernel Design
**Owner: Bhanuja** | Milestone: M2 | Phase 1

> This file is a placeholder.  Bhanuja will fill it with the
> A100 shared-memory sizing analysis and thread layout diagram.

## Sections to complete

- [ ] A100 shared-memory budget (164 KB/SM)
- [ ] Tile-size selection for Q + K + V at d_head=64
- [ ] Thread-to-data mapping (threads per block, warps)
- [ ] Shared-memory layout diagram (bank-conflict-free indexing)
- [ ] Byte calculation: input load / QKV compute / attention output
