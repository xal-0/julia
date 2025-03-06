module SyntaxUtil

import Base.JuliaSyntax: build_tree
using Base.JuliaSyntax:
    AbstractSyntaxData, GreenNode, ParseStream, SourceFile, SyntaxHead, SyntaxNode, TreeNode,
    _unsafe_wrap_substring, byte_range, children, first_byte, is_leaf, is_trivia, kind, span

export CursorNode, char_range, char_last, children_nt, find_delim, seek_pos

# Similar to SyntaxNode, but keeps trivia, tracks each child's location in its
# parent, and doesn't parse literals.
# Code extracted from JuliaSyntax/src/syntax_tree.jl
struct CursorData <: AbstractSyntaxData
    source::SourceFile
    raw::GreenNode{SyntaxHead}
    position::Int
    index::Int
end

const CursorNode = TreeNode{CursorData}

function CursorNode(source::SourceFile, raw::GreenNode{SyntaxHead};
                    position::Integer=1)
    GC.@preserve source begin
        raw_offset, txtbuf = _unsafe_wrap_substring(source.code)
        offset = raw_offset - source.byte_offset
        _to_CursorNode(source, txtbuf, offset, raw, convert(Int, position))
    end
end

function _to_CursorNode(source::SourceFile, txtbuf::Vector{UInt8}, offset::Int,
                        raw::GreenNode{SyntaxHead},
                        position::Int, index::Int=-1)
    if is_leaf(raw)
        return CursorNode(nothing, nothing, CursorData(source, raw, position, index))
    else
        cs = CursorNode[]
        pos = position
        for (i,rawchild) in enumerate(children(raw))
            push!(cs, _to_CursorNode(source, txtbuf, offset, rawchild, pos, i))
            pos += Int(rawchild.span)
        end
        node = CursorNode(nothing, cs, CursorData(source, raw, position, index))
        for c in cs
            c.parent = node
        end
        return node
    end
end

function build_tree(::Type{CursorNode}, stream::ParseStream;
                    filename=nothing, first_line=1, kws...)
    green_tree = build_tree(GreenNode, stream; kws...)
    source = SourceFile(stream, filename=filename, first_line=first_line)
    CursorNode(source, green_tree, position=first_byte(stream))
end

Base.show(io::IO, node::CursorNode) = show(io, MIME("text/plain"), node.raw)
Base.show(io::IO, mime::MIME{Symbol("text/plain")}, node::CursorNode) = show(io, mime, node.raw)

Base.Expr(node::CursorNode) =
    Expr(SyntaxNode(SourceFile(node.source[byte_range(node)]), node.raw))

char_range(node) = node.position:char_last(node)
char_last(node) = thisind(node.source, node.position + span(node) - 1)

children_nt(node) = collect(filter(!is_trivia, children(node)))

function seek_pos(node, pos)
    pos in byte_range(node) || return nothing
    (cs = children(node)) === nothing && return node
    for n in cs
        c = seek_pos(n, pos)
        c === nothing || return c
    end
    node
end

function find_parent(node, k)
    while node !== nothing && kind(node) !== k
        node = node.parent
    end
    node
end

# Return the character range between left_kind and right_kind in node.  The left
# delimiter must be present, while the range will extend to the rest of the node
# if the right delimiter is missing.
function find_delim(node, left_kind, right_kind)
    cs = children(node)
    left = first(c.position for c in cs if kind(c) == left_kind)
    left = nextind(node.source, left)
    right = findlast(c -> kind(c) == right_kind, cs)
    found = right !== nothing
    right = found ? char_last(cs[right-1]) : char_last(node)
    return left:right, found
end

end
