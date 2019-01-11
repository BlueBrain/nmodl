#include "visitors/json_visitor.hpp"

{% for node in nodes %}
void JSONVisitor::visit_{{ node.class_name|snake_case }}({{ node.class_name }}* node) {
    {% if node.has_children() %}
    printer->push_block(node->get_node_type_name());
    node->visit_children(this);
    {% if node.is_data_type_node() %}
            {% if node.is_integer_node() %}
    if(!node->get_macro_name()) {
        std::stringstream ss;
        ss << node->eval();
        printer->add_node(ss.str());
    }
            {% else %}
    std::stringstream ss;
    ss << node->eval();
    printer->add_node(ss.str());
            {% endif %}
        {% endif %}
    printer->pop_block();
        {% if node.is_program_node() %}
    flush();
        {% endif %}
    {% else %}
    (void)node;
    printer->add_node("{{ node.class_name }}");
    {% endif %}
}

{% endfor %}
