# GPU particle storage as one n × (3D + 2) matrix (position, velocity, best_position, cost,
# best_cost per row), so a warp reading one field hits consecutive addresses.
struct SoAParticles{T1, T2, M <: AbstractMatrix{T2}} <: AbstractVector{SPSOParticle{T1, T2}}
    data::M
end
SoAParticles{T1}(data::AbstractMatrix{T2}) where {T1, T2} = SoAParticles{T1, T2, typeof(data)}(data)
Base.size(p::SoAParticles) = (size(p.data, 1),)

@inline function Base.getindex(p::SoAParticles{T1}, i::Int) where {T1}
    d, D = p.data, length(T1)
    @inbounds return SPSOParticle(
        StaticArrays.sacollect(T1, d[i, j] for j in 1:D),
        StaticArrays.sacollect(T1, d[i, D + j] for j in 1:D), d[i, 3D + 1],
        StaticArrays.sacollect(T1, d[i, 2D + j] for j in 1:D), d[i, 3D + 2]
    )
end

@inline function Base.setindex!(p::SoAParticles{T1}, q::SPSOParticle, i::Int) where {T1}
    d, D = p.data, length(T1)
    @inbounds for j in 1:D
        d[i, j], d[i, D + j], d[i, 2D + j] = q.position[j], q.velocity[j], q.best_position[j]
    end
    @inbounds d[i, 3D + 1], d[i, 3D + 2] = q.cost, q.best_cost
    return p
end

Adapt.adapt_structure(to, p::SoAParticles{T1}) where {T1} = SoAParticles{T1}(adapt(to, p.data))
KernelAbstractions.get_backend(p::SoAParticles) = KernelAbstractions.get_backend(p.data)

particle_storage(backend, ::Type{T1}, n) where {T1} =
    KernelAbstractions.allocate(backend, SPSOParticle{T1, eltype(T1)}, n)
particle_storage(backend::KernelAbstractions.GPU, ::Type{T1}, n) where {T1} =
    SoAParticles{T1}(KernelAbstractions.allocate(backend, eltype(T1), n, 3length(T1) + 2))

# Host-side reads done on the device instead of element by element.
function Base.minimum(p::SoAParticles{T1}) where {T1}
    _, i = findmin(view(p.data, :, 3length(T1) + 2))
    return SoAParticles{T1}(Array(view(p.data, i:i, :)))[1]
end
Base.Array(p::SoAParticles{T1}) where {T1} = collect(SoAParticles{T1}(Array(p.data)))

@kernel function soa_positions!(out, particles)
    i = @index(Global, Linear)
    @inbounds out[i] = particles[i].position
end
get_positions(particles) = get_pos.(particles)
function get_positions(p::SoAParticles{T1}) where {T1}
    out = KernelAbstractions.allocate(KernelAbstractions.get_backend(p), T1, length(p))
    soa_positions!(KernelAbstractions.get_backend(p))(out, p; ndrange = length(p))
    return out
end
