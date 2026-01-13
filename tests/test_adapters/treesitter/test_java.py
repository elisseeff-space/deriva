"""Tests for Java language extractor."""

from __future__ import annotations

from deriva.adapters.treesitter import TreeSitterManager


class TestJavaTypes:
    """Tests for Java type extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_extracts_class(self):
        """Should extract Java class."""
        source = """
public class UserService {
    private String name;
}
"""
        types = self.manager.extract_types(source, language="java")

        assert len(types) >= 1
        service = next((t for t in types if t.name == "UserService"), None)
        assert service is not None
        assert service.kind == "class"

    def test_extracts_class_with_extends(self):
        """Should extract class inheritance."""
        source = """
public class Admin extends User {
    private int level;
}
"""
        types = self.manager.extract_types(source, language="java")

        admin = next((t for t in types if t.name == "Admin"), None)
        assert admin is not None

    def test_extracts_class_with_implements(self):
        """Should extract interface implementation."""
        source = """
public class UserServiceImpl implements UserService, Serializable {
    public void save(User user) {}
}
"""
        types = self.manager.extract_types(source, language="java")

        impl = next((t for t in types if t.name == "UserServiceImpl"), None)
        assert impl is not None

    def test_extracts_interface(self):
        """Should extract Java interface."""
        source = """
public interface Repository<T> {
    T findById(Long id);
    void save(T entity);
}
"""
        types = self.manager.extract_types(source, language="java")

        repo = next((t for t in types if t.name == "Repository"), None)
        assert repo is not None
        assert repo.kind == "interface"

    def test_extracts_enum(self):
        """Should extract Java enum."""
        source = """
public enum Status {
    PENDING,
    ACTIVE,
    COMPLETED
}
"""
        types = self.manager.extract_types(source, language="java")

        status = next((t for t in types if t.name == "Status"), None)
        assert status is not None

    def test_extracts_abstract_class(self):
        """Should extract abstract class."""
        source = """
public abstract class BaseEntity {
    protected Long id;

    public abstract void validate();
}
"""
        types = self.manager.extract_types(source, language="java")

        entity = next((t for t in types if t.name == "BaseEntity"), None)
        assert entity is not None

    def test_extracts_inner_class(self):
        """Should handle inner classes."""
        source = """
public class Outer {
    public class Inner {
        private int value;
    }
}
"""
        types = self.manager.extract_types(source, language="java")

        # Should extract at least outer class
        assert len(types) >= 1


class TestJavaMethods:
    """Tests for Java method extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_extracts_public_method(self):
        """Should extract public method."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        methods = self.manager.extract_methods(source, language="java")

        add = next((m for m in methods if m.name == "add"), None)
        assert add is not None

    def test_extracts_private_method(self):
        """Should extract private method."""
        source = """
public class Service {
    private void helper() {
        // internal logic
    }
}
"""
        methods = self.manager.extract_methods(source, language="java")

        helper = next((m for m in methods if m.name == "helper"), None)
        assert helper is not None

    def test_extracts_static_method(self):
        """Should extract static method."""
        source = """
public class Utils {
    public static String format(String input) {
        return input.trim();
    }
}
"""
        methods = self.manager.extract_methods(source, language="java")

        format_method = next((m for m in methods if m.name == "format"), None)
        assert format_method is not None

    def test_extracts_method_with_generics(self):
        """Should handle methods with generics."""
        source = """
public class Repository {
    public <T> List<T> findAll(Class<T> type) {
        return new ArrayList<>();
    }
}
"""
        methods = self.manager.extract_methods(source, language="java")

        find_all = next((m for m in methods if m.name == "findAll"), None)
        assert find_all is not None

    def test_extracts_constructor(self):
        """Should extract constructor."""
        source = """
public class Person {
    private String name;

    public Person(String name) {
        this.name = name;
    }
}
"""
        methods = self.manager.extract_methods(source, language="java")
        constructor = next((m for m in methods if m.name == "Person"), None)

        assert constructor is not None
        assert constructor.class_name == "Person"

    def test_extracts_overloaded_methods(self):
        """Should extract overloaded methods."""
        source = """
public class Printer {
    public void print(String s) {}
    public void print(int i) {}
    public void print(String s, int count) {}
}
"""
        methods = self.manager.extract_methods(source, language="java")

        print_methods = [m for m in methods if m.name == "print"]
        assert len(print_methods) == 3


class TestJavaImports:
    """Tests for Java import extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_extracts_single_import(self):
        """Should extract single class import."""
        source = """
import java.util.List;
"""
        imports = self.manager.extract_imports(source, language="java")

        assert len(imports) == 1

    def test_extracts_wildcard_import(self):
        """Should extract wildcard import."""
        source = """
import java.util.*;
"""
        imports = self.manager.extract_imports(source, language="java")

        assert len(imports) == 1

    def test_extracts_static_import(self):
        """Should extract static import."""
        source = """
import static java.lang.Math.PI;
import static java.lang.Math.*;
"""
        imports = self.manager.extract_imports(source, language="java")

        assert len(imports) >= 1

    def test_extracts_multiple_imports(self):
        """Should extract multiple imports."""
        source = """
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.io.IOException;
"""
        imports = self.manager.extract_imports(source, language="java")

        assert len(imports) == 4


class TestJavaEdgeCases:
    """Tests for edge cases in Java extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_handles_empty_file(self):
        """Should handle empty file."""
        types = self.manager.extract_types("", language="java")
        assert types == []

    def test_handles_annotations(self):
        """Should handle annotated classes."""
        source = """
@Entity
@Table(name = "users")
public class User {
    @Id
    private Long id;
}
"""
        types = self.manager.extract_types(source, language="java")

        user = next((t for t in types if t.name == "User"), None)
        assert user is not None

    def test_handles_package_declaration(self):
        """Should handle package declaration."""
        source = """
package com.example.service;

public class MyService {
}
"""
        types = self.manager.extract_types(source, language="java")

        assert len(types) >= 1

    def test_handles_record(self):
        """Should handle Java records."""
        source = """
public record User(String name, int age) {
}
"""
        types = self.manager.extract_types(source, language="java")
        user = next((t for t in types if t.name == "User"), None)

        assert user is not None
